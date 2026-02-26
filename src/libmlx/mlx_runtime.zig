const std = @import("std");
const utils = @import("utils.zig");
const core = @import("mlx_core.zig");
const nn = @import("mlx_nn.zig");

const C = core.C;
const Array = core.Array;
const INT32 = core.INT32;
const UINT32 = core.UINT32;
const FLOAT16 = core.FLOAT16;
const BFLOAT16 = core.BFLOAT16;

const MLXConfig = nn.MLXConfig;
const Module = nn.Module;
const Embedding = nn.Embedding;
const RMSNorm = nn.RMSNorm;
const Linear = nn.Linear;
const QuantConfig = nn.QuantConfig;

const now_ms = core.now_ms;
const load_model_safetensors = core.load_model_safetensors;
const create_causal_mask = core.create_causal_mask;
const array_new_data = core.array_new_data;
const array_set_data = core.array_set_data;
const array_new = core.array_new;
const array_set = core.array_set;
const array_free = core.array_free;
const array_dim = core.array_dim;
const all_min = core.all_min;
const arange = core.arange;
const argsort = core.argsort;
const cumsum = core.cumsum;
const expand_dims = core.expand_dims;
const greater_equal = core.greater_equal;
const logical_or = core.logical_or;
const multiply = core.multiply;
const random_categorical = core.random_categorical;
const reshape = core.reshape;
const softmax = core.softmax;
const take = core.take;
const take_along_axis = core.take_along_axis;
const topk = core.topk;
const argmax = core.argmax;
const item = core.item;
const float = core.float;
const int = core.int;
const less_equal = core.less_equal;
const where = core.where;

inline fn discard(_: anytype) void {}

pub const ModelConfig = struct {
    bos_token_id: c_int,
    eos_token_id: u32,
    hidden_size: c_int,
    num_hidden_layers: c_int,
    intermediate_size: c_int,
    num_attention_heads: c_int,
    rms_norm_eps: f32,
    vocab_size: c_int,
    rope_theta: f32,
    max_position_embeddings: c_int,
    tie_word_embeddings: bool,
    torch_dtype: []u8,
    rope_traditional: bool = false,
    eos_token_ids: ?[]u32 = null,
    num_key_value_heads: ?c_int = null,
    quantization: ?QuantConfig = null,
};

pub const SamplingConfig = struct {
    temperature: f32 = 0.0,
    top_p: f32 = 1.0,
    top_k: usize = 0,

    pub fn validate(self: SamplingConfig) !void {
        if (!std.math.isFinite(self.temperature) or self.temperature < 0.0) {
            return error.InvalidSamplingConfig;
        }
        if (!std.math.isFinite(self.top_p) or self.top_p <= 0.0 or self.top_p > 1.0) {
            return error.InvalidSamplingConfig;
        }
    }
};

fn model_init_impl(
    comptime BlockType: type,
    comptime ConfigType: type,
    comptime Self: type,
    model_config: ConfigType,
    mlx_config: *MLXConfig,
) !*Self {
    if (model_config.num_hidden_layers <= 0) {
        return error.InvalidModelConfig;
    }
    if (model_config.vocab_size <= 0) {
        return error.InvalidModelConfig;
    }
    std.debug.assert(model_config.num_hidden_layers > 0);
    std.debug.assert(model_config.vocab_size > 0);

    const self = try mlx_config.allocator.create(Self);
    self.* = .{
        .base = Module.init(mlx_config.allocator, mlx_config.stream),
        .embed_tokens = undefined,
        .layers = undefined,
        .norm = undefined,
        .tie_word_embeddings = model_config.tie_word_embeddings,
        .lm_head = undefined,
        .config = model_config,
    };
    self.embed_tokens = try Embedding.init(
        "model.embed_tokens",
        model_config.quantization,
        mlx_config,
    );
    self.norm = try RMSNorm.init("model.norm", model_config.rms_norm_eps, mlx_config);
    self.layers = try mlx_config.allocator.alloc(
        Self.LayerRef,
        @intCast(model_config.num_hidden_layers),
    );
    for (0..@intCast(model_config.num_hidden_layers)) |i| {
        const i_key = try self.base.alloc_join("model.layers", i);
        self.layers[i] = .{ .ptr = try BlockType.init(i_key, model_config, mlx_config) };
    }
    if (model_config.tie_word_embeddings) {
        self.lm_head = null;
    } else {
        self.lm_head = try Linear.init(
            "lm_head",
            false,
            model_config.quantization,
            mlx_config,
        );
    }
    return self;
}

fn model_deinit_impl(comptime Self: type, self: *Self) void {
    self.embed_tokens.deinit();
    for (self.layers) |layer_ref| {
        layer_ref.ptr.deinit();
    }
    self.base.allocator.free(self.layers);
    self.norm.deinit();
    if (self.tie_word_embeddings) {
        std.debug.assert(self.lm_head == null);
    } else if (self.lm_head != null) {
        self.lm_head.?.deinit();
    }
    self.base.deinit();
    self.base.allocator.destroy(self);
}

fn model_cache_offset(cache: ?*Cache) c_int {
    if (cache) |c| {
        return c.offset;
    }
    std.debug.assert(cache == null);
    return 0;
}

fn model_layer_cache(cache: ?*Cache, idx: u32) ?*KVCache {
    const idx_usize: usize = @intCast(idx);
    if (cache) |c| {
        if (idx_usize >= c.layers.len) {
            return null;
        }
        std.debug.assert(idx_usize < c.layers.len);
        return &c.layers[idx_usize];
    }
    std.debug.assert(cache == null);
    return null;
}

fn transformer_init_impl(
    comptime ModelType: type,
    comptime ConfigType: type,
    allocator: std.mem.Allocator,
    model_path: []const u8,
) !struct {
    mlx_config: MLXConfig,
    model: *ModelType,
    eos_token_ids: []u32,
} {
    if (model_path.len == 0) {
        return error.InvalidModelPath;
    }
    std.debug.assert(model_path.len > 0);

    const model_config = try utils.load_config_json(
        ConfigType,
        allocator,
        model_path,
        true,
    );
    defer model_config.deinit();
    var mlx_dtype: C.mlx_dtype = FLOAT16;
    if (std.mem.eql(u8, "bfloat16", model_config.value.torch_dtype)) {
        mlx_dtype = BFLOAT16;
    }
    var mlx_config = try MLXConfig.init(allocator, mlx_dtype);
    const model = try ModelType.init(model_config.value, &mlx_config);
    try load_model_safetensors(&mlx_config.weights_hash, model_path, mlx_config.stream);
    return .{
        .mlx_config = mlx_config,
        .model = model,
        .eos_token_ids = model_config.value.eos_token_ids.?,
    };
}

const GenerateStepResult = struct {
    token: u32,
    should_stop: bool,
};

const GenerateMetrics = struct {
    prompt_ms: f16,
    generation_start_ms: i128,
};

fn transformer_print_generation_stats(
    initial_tokens: []const u32,
    generated_count: u32,
    metrics: GenerateMetrics,
) void {
    if (initial_tokens.len == 0) {
        return;
    }
    if (metrics.prompt_ms <= 0) {
        return;
    }
    std.debug.assert(initial_tokens.len > 0);
    std.debug.assert(metrics.prompt_ms > 0);

    const prompt_tps = @as(f16, @floatFromInt(initial_tokens.len)) / (metrics.prompt_ms / 1000.0);
    std.debug.print(
        "\nPrompt:     {d:.2} tokens-per-second ({d} tokens in {d:.2} ms)\n",
        .{ prompt_tps, initial_tokens.len, metrics.prompt_ms },
    );
    if (generated_count > 0) {
        const end_time = now_ms();
        const gen_ms = @as(f16, @floatFromInt(end_time - metrics.generation_start_ms));
        const gen_tps = @as(f16, @floatFromInt(generated_count)) / (gen_ms / 1000.0);
        std.debug.print(
            "Generation: {d:.2} tokens-per-second ({d} tokens in {d:.2} ms)\n",
            .{ gen_tps, generated_count, gen_ms },
        );
    } else {
        std.debug.assert(generated_count == 0);
    }
}

const GenerateTokensResult = struct {
    generated_count: u32,
    metrics: GenerateMetrics,
};

fn transformer_greedy_token(
    comptime TransformerType: type,
    transformer: *TransformerType,
    logits: Array,
) !u32 {
    var greedy = array_new();
    defer array_free(greedy);
    try argmax(&greedy, logits, 1, false, transformer.mlx_config.stream);

    var token: u32 = 0;
    try item(&token, greedy);
    return token;
}

fn transformer_apply_temperature(
    comptime TransformerType: type,
    transformer: *TransformerType,
    logits: *Array,
    temperature: f32,
) !void {
    if (temperature == 1.0) return;
    if (temperature <= 0.0) return error.InvalidSamplingConfig;
    std.debug.assert(temperature > 0.0);

    var scaled_logits = array_new();
    defer array_free(scaled_logits);
    try multiply(&scaled_logits, logits.*, float(1.0 / temperature), transformer.mlx_config.stream);
    try array_set(logits, scaled_logits);
}

fn transformer_apply_top_k(
    comptime TransformerType: type,
    transformer: *TransformerType,
    logits: *Array,
    top_k_value: u32,
) !void {
    if (top_k_value == 0) return;

    const vocab_size = array_dim(logits.*, 1);
    if (vocab_size <= 0) return error.InvalidInput;
    std.debug.assert(vocab_size > 0);
    const vocab_size_u32: u32 = @intCast(vocab_size);
    if (top_k_value >= vocab_size_u32) return;
    if (top_k_value > std.math.maxInt(c_int)) return error.InvalidSamplingConfig;
    std.debug.assert(top_k_value < vocab_size_u32);
    std.debug.assert(top_k_value > 0);

    var top_values = array_new();
    var min_top_value = array_new();
    var keep_mask = array_new();
    var filtered_logits = array_new();
    defer {
        array_free(top_values);
        array_free(min_top_value);
        array_free(keep_mask);
        array_free(filtered_logits);
    }

    try topk(&top_values, logits.*, @intCast(top_k_value), 1, transformer.mlx_config.stream);
    try all_min(&min_top_value, top_values, false, transformer.mlx_config.stream);
    try greater_equal(&keep_mask, logits.*, min_top_value, transformer.mlx_config.stream);
    try where(
        &filtered_logits,
        keep_mask,
        logits.*,
        float(-std.math.inf(f32)),
        transformer.mlx_config.stream,
    );
    try array_set(logits, filtered_logits);
}

fn transformer_first_position_mask(
    comptime TransformerType: type,
    transformer: *TransformerType,
    vocab_size: c_int,
    result: *Array,
) !void {
    var position_ids = array_new();
    var position_ids_2d = array_new();
    defer {
        array_free(position_ids);
        array_free(position_ids_2d);
    }

    try arange(
        &position_ids,
        0,
        @floatFromInt(vocab_size),
        1,
        INT32,
        transformer.mlx_config.stream,
    );
    try expand_dims(&position_ids_2d, position_ids, &[_]c_int{0}, transformer.mlx_config.stream);
    try less_equal(result, position_ids_2d, int(0), transformer.mlx_config.stream);
}

fn transformer_sample_top_p_token(
    comptime TransformerType: type,
    transformer: *TransformerType,
    working_logits: Array,
    top_p_value: f32,
) !u32 {
    var neg_logits = array_new();
    defer array_free(neg_logits);
    try multiply(&neg_logits, working_logits, float(-1.0), transformer.mlx_config.stream);

    var sorted_indices = array_new();
    defer array_free(sorted_indices);
    try argsort(&sorted_indices, neg_logits, 1, transformer.mlx_config.stream);

    var sorted_logits = array_new();
    defer array_free(sorted_logits);
    try take_along_axis(
        &sorted_logits,
        working_logits,
        sorted_indices,
        1,
        transformer.mlx_config.stream,
    );

    var sorted_probs = array_new();
    defer array_free(sorted_probs);
    try softmax(
        &sorted_probs,
        sorted_logits,
        &[_]c_int{1},
        true,
        transformer.mlx_config.stream,
    );

    var cumulative_probs = array_new();
    defer array_free(cumulative_probs);
    try cumsum(&cumulative_probs, sorted_probs, 1, false, true, transformer.mlx_config.stream);

    var keep_mask_sorted = array_new();
    defer array_free(keep_mask_sorted);
    try less_equal(
        &keep_mask_sorted,
        cumulative_probs,
        float(top_p_value),
        transformer.mlx_config.stream,
    );

    const vocab_size = array_dim(sorted_logits, 1);
    if (vocab_size <= 0) return error.InvalidInput;
    std.debug.assert(vocab_size > 0);

    var first_position_mask = array_new();
    defer array_free(first_position_mask);
    try transformer_first_position_mask(
        TransformerType,
        transformer,
        vocab_size,
        &first_position_mask,
    );
    try logical_or(
        &keep_mask_sorted,
        keep_mask_sorted,
        first_position_mask,
        transformer.mlx_config.stream,
    );

    var filtered_sorted_logits = array_new();
    defer array_free(filtered_sorted_logits);
    try where(
        &filtered_sorted_logits,
        keep_mask_sorted,
        sorted_logits,
        float(-std.math.inf(f32)),
        transformer.mlx_config.stream,
    );

    return transformer_sample_sorted_token(
        TransformerType,
        transformer,
        sorted_indices,
        filtered_sorted_logits,
    );
}

fn transformer_sample_sorted_token(
    comptime TransformerType: type,
    transformer: *TransformerType,
    sorted_indices: Array,
    filtered_sorted_logits: Array,
) !u32 {
    if (sorted_indices.ctx == null) return error.InvalidInput;
    if (filtered_sorted_logits.ctx == null) return error.InvalidInput;
    std.debug.assert(sorted_indices.ctx != null);
    std.debug.assert(filtered_sorted_logits.ctx != null);

    var sampled_sorted_index = array_new();
    var sampled_sorted_index_2d = array_new();
    var sampled_token_2d = array_new();
    var sampled_token = array_new();
    defer {
        array_free(sampled_sorted_index);
        array_free(sampled_sorted_index_2d);
        array_free(sampled_token_2d);
        array_free(sampled_token);
    }

    try random_categorical(
        &sampled_sorted_index,
        filtered_sorted_logits,
        1,
        transformer.mlx_config.stream,
    );
    try reshape(
        &sampled_sorted_index_2d,
        sampled_sorted_index,
        &[_]c_int{ 1, 1 },
        transformer.mlx_config.stream,
    );
    try take_along_axis(
        &sampled_token_2d,
        sorted_indices,
        sampled_sorted_index_2d,
        1,
        transformer.mlx_config.stream,
    );
    try reshape(&sampled_token, sampled_token_2d, &[_]c_int{1}, transformer.mlx_config.stream);

    var token: u32 = 0;
    try item(&token, sampled_token);
    return token;
}

fn transformer_sample_token(
    comptime TransformerType: type,
    transformer: *TransformerType,
    logits: Array,
    sampling: SamplingConfig,
) !u32 {
    if (sampling.temperature == 0.0) {
        return transformer_greedy_token(TransformerType, transformer, logits);
    }
    if (sampling.top_k > std.math.maxInt(u32)) return error.InvalidSamplingConfig;
    std.debug.assert(sampling.temperature > 0.0);

    var working_logits = array_new();
    defer array_free(working_logits);
    try array_set(&working_logits, logits);

    try transformer_apply_temperature(
        TransformerType,
        transformer,
        &working_logits,
        sampling.temperature,
    );
    try transformer_apply_top_k(
        TransformerType,
        transformer,
        &working_logits,
        @intCast(sampling.top_k),
    );

    if (sampling.top_p < 1.0) {
        return transformer_sample_top_p_token(
            TransformerType,
            transformer,
            working_logits,
            sampling.top_p,
        );
    }

    var sampled_token = array_new();
    defer array_free(sampled_token);
    try random_categorical(&sampled_token, working_logits, 1, transformer.mlx_config.stream);

    var token: u32 = 0;
    try item(&token, sampled_token);
    return token;
}

fn transformer_generate_next_token(
    comptime TransformerType: type,
    transformer: *TransformerType,
    cache: *Cache,
    toks: Array,
    logits: Array,
    mask: Array,
    sampling: SamplingConfig,
) !struct { token: u32, logits: Array, mask: Array } {
    var logits_ = logits;
    var mask_ = mask;
    try create_causal_mask(
        &mask_,
        array_dim(toks, 1),
        cache.offset,
        transformer.mlx_config.dtype,
        transformer.mlx_config.stream,
    );
    try transformer.model.forward(&logits_, toks, mask_, cache);
    try take(&logits_, logits_, int(-1), 1, transformer.mlx_config.stream);
    const token = try transformer_sample_token(TransformerType, transformer, logits_, sampling);
    return .{ .token = token, .logits = logits_, .mask = mask_ };
}

fn transformer_commit_generated_token(
    toks: *Array,
    output_tokens: []u32,
    index: u32,
    token: u32,
) !void {
    const index_usize: usize = @intCast(index);
    if (index_usize >= output_tokens.len) return error.InvalidInput;
    std.debug.assert(index_usize < output_tokens.len);
    output_tokens[index_usize] = token;
    try array_set_data(toks, &output_tokens[index_usize], &[_]c_int{ 1, 1 }, UINT32);
}

fn transformer_generate_tokens(
    comptime TransformerType: type,
    transformer: *TransformerType,
    output_tokens: []u32,
    cache: *Cache,
    toks: Array,
    logits: Array,
    mask: Array,
    start_ms: i128,
    sampling: SamplingConfig,
) !GenerateTokensResult {
    if (output_tokens.len == 0) return error.InvalidInput;
    if (cache.layers.len == 0) return error.InvalidInput;
    if (array_dim(toks, 1) <= 0) return error.InvalidInput;
    if (toks.ctx == null) return error.InvalidInput;
    if (logits.ctx == null) return error.InvalidInput;
    if (mask.ctx == null) return error.InvalidInput;
    std.debug.assert(output_tokens.len > 0);
    std.debug.assert(cache.layers.len > 0);
    std.debug.assert(array_dim(toks, 1) > 0);
    std.debug.assert(toks.ctx != null);
    std.debug.assert(logits.ctx != null);
    std.debug.assert(mask.ctx != null);

    var logits_ = logits;
    var mask_ = mask;
    var toks_ = toks;
    var metrics = GenerateMetrics{ .prompt_ms = 0, .generation_start_ms = start_ms };

    const generated_count: u32 = generation: for (output_tokens, 0..) |_, index| {
        const token = try transformer_generate_next_token(
            TransformerType,
            transformer,
            cache,
            toks_,
            logits_,
            mask_,
            sampling,
        );
        logits_ = token.logits;
        mask_ = token.mask;

        try transformer_commit_generated_token(&toks_, output_tokens, @intCast(index), token.token);
        std.debug.print(
            "Generated token {d}/{d}: {d}\n",
            .{ index + 1, output_tokens.len, token.token },
        );

        if (index == 0) {
            const current_time = now_ms();
            metrics.prompt_ms = @floatFromInt(current_time - start_ms);
            metrics.generation_start_ms = current_time;
        }
        if (std.mem.indexOfScalar(u32, transformer.eos_token_ids, token.token) != null) {
            break :generation @intCast(index + 1);
        }
    } else @intCast(output_tokens.len);

    return .{ .generated_count = generated_count, .metrics = metrics };
}

pub fn Model(comptime BlockType: type, comptime ConfigType: type) type {
    return struct {
        const Self = @This();
        const LayerRef = struct {
            ptr: *BlockType,
        };

        base: Module,
        embed_tokens: *Embedding,
        norm: *RMSNorm,
        layers: []LayerRef,
        tie_word_embeddings: bool,
        lm_head: ?*Linear,
        config: ConfigType,

        pub fn init(model_config: ConfigType, mlx_config: *MLXConfig) !*Self {
            return model_init_impl(BlockType, ConfigType, Self, model_config, mlx_config);
        }

        pub fn deinit(self: *Self) void {
            model_deinit_impl(Self, self);
        }

        pub fn forward(
            self: *Self,
            result: *Array,
            toks: Array,
            mask: ?Array,
            cache: ?*Cache,
        ) !void {
            if (toks.ctx == null) {
                return error.InvalidInput;
            }
            if (array_dim(toks, 1) <= 0) {
                return error.InvalidInput;
            }
            if (result.ctx == null) {
                std.debug.assert(result.ctx == null);
            } else {
                std.debug.assert(result.ctx != null);
            }
            std.debug.assert(toks.ctx != null);
            std.debug.assert(array_dim(toks, 0) == 1);
            std.debug.assert(array_dim(toks, 1) > 0);
            std.debug.assert(self.layers.len > 0);
            std.debug.assert(self.tie_word_embeddings or self.lm_head != null);
            std.debug.assert(self.tie_word_embeddings == (self.lm_head == null));
            if (cache) |c| {
                std.debug.assert(c.layers.len == self.layers.len);
            }
            if (mask) |m| {
                std.debug.assert(m.ctx != null);
                std.debug.assert(array_dim(m, 0) == 1);
                std.debug.assert(array_dim(m, 1) > 0);
            }

            var x = array_new();
            defer array_free(x);
            const offset = model_cache_offset(cache);
            try self.embed_tokens.forward(&x, toks);
            for (self.layers, 0..) |layer_ref, i| {
                if (i > std.math.maxInt(u32)) {
                    return error.InvalidCache;
                }
                const layer_cache = model_layer_cache(cache, @intCast(i));
                if (cache != null and layer_cache == null) {
                    return error.InvalidCache;
                }
                try layer_ref.ptr.forward(&x, x, mask, layer_cache, offset);
            }
            try self.norm.forward(&x, x);
            if (cache) |c| {
                const seq_len = array_dim(toks, 1);
                std.debug.assert(seq_len > 0);
                c.offset += seq_len;
            }

            if (self.tie_word_embeddings) {
                try self.embed_tokens.as_linear(result, x);
            } else {
                try self.lm_head.?.forward(result, x);
            }
        }
    };
}

fn transformer_validate_generate_request(
    transformer: anytype,
    initial_tokens: []const u32,
    num_tokens: u32,
    sampling: SamplingConfig,
) !void {
    try sampling.validate();
    if (sampling.temperature > 0.0 and sampling.top_k > std.math.maxInt(c_int)) {
        return error.InvalidSamplingConfig;
    }
    if (initial_tokens.len == 0) {
        return error.InvalidInput;
    }
    if (num_tokens == 0) {
        return error.InvalidInput;
    }
    if (transformer.eos_token_ids.len == 0) {
        return error.InvalidModelConfig;
    }
    if (transformer.model.layers.len == 0) {
        return error.InvalidModelState;
    }
    std.debug.assert(initial_tokens.len > 0);
    std.debug.assert(num_tokens > 0);
    std.debug.assert(transformer.eos_token_ids.len > 0);
    std.debug.assert(transformer.model.layers.len > 0);
}

fn transformer_run_generation(
    transformer: anytype,
    initial_tokens: []const u32,
    output_tokens: []u32,
    sampling: SamplingConfig,
) !GenerateTokensResult {
    if (initial_tokens.len == 0) return error.InvalidInput;
    if (output_tokens.len == 0) return error.InvalidInput;
    if (transformer.model.layers.len == 0) return error.InvalidModelState;
    std.debug.assert(initial_tokens.len > 0);
    std.debug.assert(output_tokens.len > 0);
    std.debug.assert(transformer.model.layers.len > 0);

    var cache = try Cache.init(transformer.mlx_config.allocator, transformer.model.layers.len, 2);
    defer cache.deinit();

    const toks = try array_new_data(
        initial_tokens.ptr,
        &[_]c_int{ 1, @intCast(initial_tokens.len) },
        UINT32,
    );
    const logits = array_new();
    const mask = array_new();
    defer {
        array_free(toks);
        array_free(logits);
        array_free(mask);
    }

    return transformer_generate_tokens(
        @TypeOf(transformer.*),
        transformer,
        output_tokens,
        &cache,
        toks,
        logits,
        mask,
        now_ms(),
        sampling,
    );
}

const GenerateOutput = struct {
    output_tokens: []u32,
    generated_count: u32,
    metrics: GenerateMetrics,
};

fn transformer_generate_output(
    transformer: anytype,
    initial_tokens: []const u32,
    num_tokens: u32,
    sampling: SamplingConfig,
) !GenerateOutput {
    if (initial_tokens.len == 0) return error.InvalidInput;
    if (num_tokens == 0) return error.InvalidInput;
    if (transformer.model.layers.len == 0) return error.InvalidModelState;
    std.debug.assert(initial_tokens.len > 0);
    std.debug.assert(num_tokens > 0);
    std.debug.assert(transformer.model.layers.len > 0);

    var output_tokens = try transformer.mlx_config.allocator.alloc(u32, @intCast(num_tokens));
    errdefer transformer.mlx_config.allocator.free(output_tokens);

    const generation = try transformer_run_generation(
        transformer,
        initial_tokens,
        output_tokens,
        sampling,
    );
    if (generation.generated_count < num_tokens) {
        output_tokens = try transformer.mlx_config.allocator.realloc(
            output_tokens,
            @intCast(generation.generated_count),
        );
    }
    return .{
        .output_tokens = output_tokens,
        .generated_count = generation.generated_count,
        .metrics = generation.metrics,
    };
}

fn transformer_generate_with_sampling_impl(
    transformer: anytype,
    initial_tokens: []const u32,
    num_tokens: u32,
    sampling: SamplingConfig,
) ![]u32 {
    if (initial_tokens.len == 0) return error.InvalidInput;
    if (num_tokens == 0) return error.InvalidInput;
    if (transformer.model.layers.len == 0) return error.InvalidModelState;
    if (transformer.eos_token_ids.len == 0) return error.InvalidModelConfig;
    std.debug.assert(initial_tokens.len > 0);
    std.debug.assert(num_tokens > 0);
    std.debug.assert(transformer.model.layers.len > 0);
    std.debug.assert(transformer.eos_token_ids.len > 0);

    const generation_output = try transformer_generate_output(
        transformer,
        initial_tokens,
        num_tokens,
        sampling,
    );
    return generation_output.output_tokens;
}

pub fn Transformer(comptime ModelType: type, comptime ConfigType: type) type {
    return struct {
        const Self = @This();

        mlx_config: MLXConfig,
        model: *ModelType,
        eos_token_ids: []u32,

        pub fn init(allocator: std.mem.Allocator, model_path: []const u8) !Self {
            const value = try transformer_init_impl(ModelType, ConfigType, allocator, model_path);
            return .{
                .mlx_config = value.mlx_config,
                .model = value.model,
                .eos_token_ids = value.eos_token_ids,
            };
        }

        pub fn deinit(self: *Self) void {
            self.model.deinit();
            self.mlx_config.allocator.free(self.eos_token_ids);
            self.mlx_config.deinit();
        }

        pub fn generate(self: *Self, initial_tokens: []const u32, num_tokens: usize) ![]u32 {
            if (num_tokens > std.math.maxInt(u32)) return error.InvalidInput;
            try transformer_validate_generate_request(
                self,
                initial_tokens,
                @intCast(num_tokens),
                .{},
            );
            return transformer_generate_with_sampling_impl(
                self,
                initial_tokens,
                @intCast(num_tokens),
                .{},
            );
        }

        pub fn generateWithSampling(
            self: *Self,
            initial_tokens: []const u32,
            num_tokens: usize,
            sampling: SamplingConfig,
        ) ![]u32 {
            if (num_tokens > std.math.maxInt(u32)) return error.InvalidInput;
            try transformer_validate_generate_request(
                self,
                initial_tokens,
                @intCast(num_tokens),
                sampling,
            );
            return transformer_generate_with_sampling_impl(
                self,
                initial_tokens,
                @intCast(num_tokens),
                sampling,
            );
        }
    };
}

pub const KVCache = struct {
    const Self = @This();

    k: Array,
    v: Array,
    axis: c_int,
    is_empty: bool = true,

    pub fn init(axis: c_int) Self {
        return Self{ .k = C.mlx_array_new(), .v = C.mlx_array_new(), .axis = axis };
    }

    fn slice_cache(self: *Self, offset: c_int, stream: core.Stream) core.MLXError!void {
        if (offset >= array_dim(self.k, self.axis)) return;
        const ndim = C.mlx_array_ndim(self.k);
        const start = [_]c_int{ 0, 0, 0, 0 };
        const strides = [_]c_int{ 1, 1, 1, 1 };
        var stop = [_]c_int{ 0, 0, 0, 0 };
        for (0..ndim) |idx| stop[idx] = array_dim(self.k, @intCast(idx));
        stop[@intCast(self.axis)] = offset;
        try core.op_status(
            C.mlx_slice(&self.k, self.k, &start, ndim, &stop, ndim, &strides, ndim, stream),
        );
        try core.op_status(
            C.mlx_slice(&self.k, self.v, &start, ndim, &stop, ndim, &strides, ndim, stream),
        );
        std.debug.print("Cache offset set to {d}\n", .{offset});
    }

    pub fn update(
        self: *Self,
        k: *Array,
        v: *Array,
        offset: ?c_int,
        stream: core.Stream,
    ) core.MLXError!void {
        var offset_: c_int = 0;
        if (offset) |o| {
            offset_ = o;
        } else if (self.is_empty) {
            offset_ = 0;
        } else {
            offset_ = array_dim(self.k, self.axis);
        }
        if (offset_ > 0) {
            try self.slice_cache(offset_, stream);
            const k_concat = [_]Array{ self.k, k.* };
            const k_vec = C.mlx_vector_array_new_data(&k_concat[0], 2);
            defer discard(C.mlx_vector_array_free(k_vec));
            try core.op_status(C.mlx_concatenate(k, k_vec, self.axis, stream));
            const v_concat = [_]Array{ self.v, v.* };
            const v_vec = C.mlx_vector_array_new_data(&v_concat[0], 2);
            defer discard(C.mlx_vector_array_free(v_vec));
            try core.op_status(C.mlx_concatenate(v, v_vec, self.axis, stream));
        }
        try core.array_set(&self.k, k.*);
        try core.array_set(&self.v, v.*);
        self.is_empty = false;
    }

    pub fn get(self: *Self, k: *Array, v: *Array) core.MLXError!void {
        try core.array_set(k, self.k);
        try core.array_set(v, self.v);
    }

    pub fn set(self: *Self, k: *Array, v: *Array) core.MLXError!void {
        try core.array_set(&self.k, k.*);
        try core.array_set(&self.v, v.*);
        self.is_empty = false;
    }

    pub fn deinit(self: *Self) void {
        core.array_free(self.k);
        core.array_free(self.v);
    }
};

pub const Cache = struct {
    const Self = @This();

    layers: []KVCache,
    offset: c_int,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, num_layers: usize, axis: c_int) core.MLXError!Self {
        var layers = try allocator.alloc(KVCache, num_layers);
        for (0..num_layers) |idx| {
            layers[idx] = KVCache.init(axis);
        }
        return Self{
            .layers = layers,
            .offset = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.layers) |*cache| {
            cache.deinit();
        }
        self.allocator.free(self.layers);
    }
};
