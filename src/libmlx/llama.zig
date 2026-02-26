//! llama.zig - Llama-3.2-Instruct
//!
//! Copyright 2025 Joe

const std = @import("std");
const mlx = @import("mlx.zig");
const utils = @import("utils.zig");

pub const MLP = struct {
    const Self = @This();
    base: mlx.Module = undefined,
    gate_weight: *mlx.Linear = undefined,
    up_weight: *mlx.Linear = undefined,
    down_weight: *mlx.Linear = undefined,

    pub fn init(
        key: []const u8,
        quant_config: ?mlx.QuantConfig,
        mlx_config: *mlx.MLXConfig,
    ) !*Self {
        const self = try mlx_config.allocator.create(Self);
        self.* = .{ .base = mlx.Module.init(mlx_config.allocator, mlx_config.stream) };
        self.gate_weight = try mlx.Linear.init(
            try self.base.alloc_join(key, "gate_proj"),
            false,
            quant_config,
            mlx_config,
        );
        self.up_weight = try mlx.Linear.init(
            try self.base.alloc_join(key, "up_proj"),
            false,
            quant_config,
            mlx_config,
        );
        self.down_weight = try mlx.Linear.init(
            try self.base.alloc_join(key, "down_proj"),
            false,
            quant_config,
            mlx_config,
        );
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.gate_weight.deinit();
        self.up_weight.deinit();
        self.down_weight.deinit();
        self.base.deinit();
        self.base.allocator.destroy(self);
    }

    pub fn forward(self: *Self, result: *mlx.Array, x: mlx.Array) !void {
        var gate = mlx.array_new();
        var sigmoid = mlx.array_new();
        var up = mlx.array_new();
        defer {
            mlx.array_free(gate);
            mlx.array_free(sigmoid);
            mlx.array_free(up);
        }
        try self.gate_weight.forward(&gate, x);
        try mlx.sigmoid(&sigmoid, gate, self.base.stream);
        try mlx.multiply(&gate, gate, sigmoid, self.base.stream);
        try self.up_weight.forward(&up, x);
        try mlx.multiply(&up, gate, up, self.base.stream);
        try self.down_weight.forward(result, up);
    }
};

pub const Attention = struct {
    const Self = @This();
    base: mlx.Module = undefined,
    n_heads: c_int = undefined,
    n_kv_heads: c_int = undefined,
    head_dim: c_int = undefined,
    n_repeat: c_int = undefined,
    scale: mlx.Array = undefined,
    q_weight: *mlx.Linear = undefined,
    k_weight: *mlx.Linear = undefined,
    v_weight: *mlx.Linear = undefined,
    o_weight: *mlx.Linear = undefined,
    rope: *Llama3RoPE = undefined,

    pub fn init(key: []const u8, model_config: ModelConfig, mlx_config: *mlx.MLXConfig) !*Self {
        const self = try mlx_config.allocator.create(Self);
        self.* = .{ .base = mlx.Module.init(mlx_config.allocator, mlx_config.stream) };
        self.n_heads = model_config.num_attention_heads;
        self.n_kv_heads = model_config.num_key_value_heads orelse model_config.num_attention_heads;
        self.head_dim = @divExact(model_config.hidden_size, model_config.num_attention_heads);
        self.n_repeat = @divExact(self.n_heads, self.n_kv_heads);
        self.scale = mlx.array_new_float(1.0 / @sqrt(@as(f32, @floatFromInt(self.head_dim))));
        self.q_weight = try mlx.Linear.init(
            try self.base.alloc_join(key, "q_proj"),
            false,
            model_config.quantization,
            mlx_config,
        );
        self.k_weight = try mlx.Linear.init(
            try self.base.alloc_join(key, "k_proj"),
            false,
            model_config.quantization,
            mlx_config,
        );
        self.v_weight = try mlx.Linear.init(
            try self.base.alloc_join(key, "v_proj"),
            false,
            model_config.quantization,
            mlx_config,
        );
        self.o_weight = try mlx.Linear.init(
            try self.base.alloc_join(key, "o_proj"),
            false,
            model_config.quantization,
            mlx_config,
        );
        self.rope = try Llama3RoPE.init(
            self.head_dim,
            false,
            model_config.rope_theta,
            model_config.rope_scaling,
            mlx_config,
        );
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.q_weight.deinit();
        self.k_weight.deinit();
        self.v_weight.deinit();
        self.o_weight.deinit();
        self.rope.deinit();
        mlx.array_free(self.scale);
        self.base.deinit();
        self.base.allocator.destroy(self);
    }

    pub fn forward(
        self: *Self,
        result: *mlx.Array,
        x: mlx.Array,
        mask: ?mlx.Array,
        cache: ?*mlx.KVCache,
        offset: c_int,
    ) !void {
        var q = mlx.array_new();
        var k = mlx.array_new();
        var v = mlx.array_new();
        var w = mlx.array_new();
        defer {
            mlx.array_free(q);
            mlx.array_free(k);
            mlx.array_free(v);
            mlx.array_free(w);
        }
        try self.q_weight.forward(&q, x);
        try self.k_weight.forward(&k, x);
        try self.v_weight.forward(&v, x);
        try mlx.reshape_to_heads(&q, q, self.n_heads, self.head_dim, self.base.stream);
        try mlx.reshape_to_heads(&k, k, self.n_kv_heads, self.head_dim, self.base.stream);
        try mlx.reshape_to_heads(&v, v, self.n_kv_heads, self.head_dim, self.base.stream);
        try self.rope.forward(&q, q, offset);
        try self.rope.forward(&k, k, offset);
        try mlx.multiply(&q, q, self.scale, self.base.stream);
        if (cache) |c| try c.update(&k, &v, null, self.base.stream);
        try mlx.repeat_pattern(
            &k,
            k,
            "b h l d -> b (repeat h) l d",
            .{ .repeat = self.n_repeat },
            self.base.stream,
        );
        try mlx.repeat_pattern(
            &v,
            v,
            "b h l d -> b (repeat h) l d",
            .{ .repeat = self.n_repeat },
            self.base.stream,
        );
        try mlx.einsum(
            &w,
            &[_]mlx.Array{ q, k },
            "b h l d, b h k d -> b h l k",
            self.base.stream,
        );
        if (mask) |m| try mlx.add(&w, w, m, self.base.stream);
        try mlx.softmax(&w, w, &.{3}, true, self.base.stream);
        try mlx.einsum(
            &w,
            &[_]mlx.Array{ w, v },
            "b h l k, b h k d -> b h l d",
            self.base.stream,
        );
        try mlx.reshape_from_heads(&w, w, self.base.stream);
        try self.o_weight.forward(result, w);
    }
};

pub const Llama3RoPE = struct {
    const Self = @This();
    base: mlx.Module = undefined,
    freqs: mlx.Array = undefined,
    rope_base: mlx.OptionalFloat = undefined,
    dims: c_int = undefined,
    traditional: bool = undefined,

    pub fn init(
        dims: c_int,
        traditional: bool,
        theta: f32,
        scaling_config: ModelConfig.RopeScalingConfig,
        mlx_config: *mlx.MLXConfig,
    ) !*Self {
        const self = try mlx_config.allocator.create(Self);
        self.* = .{
            .base = mlx.Module.init(mlx_config.allocator, mlx_config.stream),
            .rope_base = mlx.OptionalFloat{ .has_value = false, .value = 0.0 },
            .freqs = mlx.array_new(),
            .dims = dims,
            .traditional = traditional,
        };
        var wavelens = mlx.array_new();
        var high_freq_mask = mlx.array_new();
        var mid_freq_mask = mlx.array_new();
        var high_freq = mlx.array_new();
        var smooth_factors = mlx.array_new();
        var mid_freq = mlx.array_new();
        defer {
            mlx.array_free(wavelens);
            mlx.array_free(high_freq_mask);
            mlx.array_free(mid_freq_mask);
            mlx.array_free(high_freq);
            mlx.array_free(smooth_factors);
            mlx.array_free(mid_freq);
        }
        try mlx.arange(&self.freqs, 0, @floatFromInt(dims), 2, mlx.FLOAT32, self.base.stream);
        try mlx.divide(&self.freqs, self.freqs, mlx.float(@floatFromInt(dims)), self.base.stream);
        try mlx.power(&self.freqs, mlx.float(theta), self.freqs, self.base.stream);
        try mlx.multiply(&wavelens, mlx.float(2.0 * std.math.pi), self.freqs, self.base.stream);
        const low_freq_bound =
            scaling_config.original_max_position_embeddings / scaling_config.low_freq_factor;
        const high_freq_bound =
            scaling_config.original_max_position_embeddings / scaling_config.high_freq_factor;
        try mlx.multiply(
            &high_freq,
            self.freqs,
            mlx.float(scaling_config.factor),
            self.base.stream,
        );
        try mlx.greater(
            &high_freq_mask,
            wavelens,
            mlx.float(low_freq_bound),
            self.base.stream,
        );
        try mlx.where(&high_freq, high_freq_mask, high_freq, self.freqs, self.base.stream);
        try mlx.less_equal(
            &mid_freq_mask,
            wavelens,
            mlx.float(high_freq_bound),
            self.base.stream,
        );
        try mlx.logical_or(&mid_freq_mask, high_freq_mask, mid_freq_mask, self.base.stream);
        try mlx.logical_not(&mid_freq_mask, mid_freq_mask, self.base.stream);
        try mlx.divide(
            &smooth_factors,
            mlx.float(scaling_config.original_max_position_embeddings),
            wavelens,
            self.base.stream,
        );
        try mlx.subtract(
            &smooth_factors,
            smooth_factors,
            mlx.float(scaling_config.low_freq_factor),
            self.base.stream,
        );
        try mlx.divide(
            &smooth_factors,
            smooth_factors,
            mlx.float(scaling_config.high_freq_factor - scaling_config.low_freq_factor),
            self.base.stream,
        );
        try mlx.subtract(&mid_freq, mlx.float(1.0), smooth_factors, self.base.stream);
        try mlx.divide(&mid_freq, mid_freq, mlx.float(scaling_config.factor), self.base.stream);
        try mlx.add(&mid_freq, mid_freq, smooth_factors, self.base.stream);
        try mlx.divide(&mid_freq, self.freqs, mid_freq, self.base.stream);
        try mlx.where(&self.freqs, high_freq_mask, high_freq, self.freqs, self.base.stream);
        try mlx.array_eval(self.freqs);
        return self;
    }

    pub fn forward(self: *Self, result: *mlx.Array, x: mlx.Array, offset: c_int) !void {
        try mlx.fast_rope(
            result,
            x,
            self.dims,
            self.traditional,
            self.rope_base,
            1.0,
            offset,
            self.freqs,
            self.base.stream,
        );
        try mlx.astype(result, result.*, mlx.BFLOAT16, self.base.stream);
    }

    pub fn deinit(self: *Self) void {
        mlx.array_free(self.freqs);
        self.base.deinit();
        self.base.allocator.destroy(self);
    }
};

pub const TransformerBlock = struct {
    const Self = @This();
    base: mlx.Module = undefined,
    self_attn: *Attention = undefined,
    mlp: *MLP = undefined,
    input_layernorm: *mlx.RMSNorm = undefined,
    post_attention_layernorm: *mlx.RMSNorm = undefined,

    pub fn init(key: []const u8, model_config: ModelConfig, mlx_config: *mlx.MLXConfig) !*Self {
        const self = try mlx_config.allocator.create(Self);
        self.* = .{ .base = mlx.Module.init(mlx_config.allocator, mlx_config.stream) };
        self.self_attn = try Attention.init(
            try self.base.alloc_join(key, "self_attn"),
            model_config,
            mlx_config,
        );
        self.mlp = try MLP.init(
            try self.base.alloc_join(key, "mlp"),
            model_config.quantization,
            mlx_config,
        );
        self.input_layernorm = try mlx.RMSNorm.init(
            try self.base.alloc_join(key, "input_layernorm"),
            model_config.rms_norm_eps,
            mlx_config,
        );
        self.post_attention_layernorm = try mlx.RMSNorm.init(
            try self.base.alloc_join(key, "post_attention_layernorm"),
            model_config.rms_norm_eps,
            mlx_config,
        );
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.self_attn.deinit();
        self.mlp.deinit();
        self.input_layernorm.deinit();
        self.post_attention_layernorm.deinit();
        self.base.deinit();
        self.base.allocator.destroy(self);
    }

    pub fn forward(
        self: *Self,
        result: *mlx.Array,
        x: mlx.Array,
        mask: ?mlx.Array,
        cache: ?*mlx.KVCache,
        offset: c_int,
    ) !void {
        var attn = mlx.array_new();
        var mlp_out = mlx.array_new();
        defer {
            mlx.array_free(attn);
            mlx.array_free(mlp_out);
        }
        try self.input_layernorm.forward(&attn, x);
        try self.self_attn.forward(&attn, attn, mask, cache, offset);
        try mlx.add(&attn, attn, x, self.base.stream);
        try self.post_attention_layernorm.forward(&mlp_out, attn);
        try self.mlp.forward(&mlp_out, mlp_out);
        try mlx.add(result, mlp_out, attn, self.base.stream);
    }
};

pub const Model = mlx.Model(TransformerBlock, ModelConfig);
pub const Transformer = mlx.Transformer(Model, ModelConfig);

test "llama.zig" {
    std.debug.print("\n=== LLAMA.ZIG ===\n\n", .{});
    const allocator = std.testing.allocator;
    const initial_tokens = [_]u32{ 9906, 1917 };
    const num_tokens_to_generate = 10;
    var transformer = try Transformer.init(allocator, "Llama-3.2-1B-Instruct-4bit");
    defer transformer.deinit();
    const generated_tokens = try transformer.generate(&initial_tokens, num_tokens_to_generate);
    defer allocator.free(generated_tokens);
    std.debug.print("\nGenerated sequence: ", .{});
    for (generated_tokens) |token| {
        std.debug.print("{d} ", .{token});
    }
    std.debug.print("\n", .{});
}

pub const ModelConfig = struct {
    eos_token_id: []u32,
    eos_token_ids: ?[]u32 = null,
    hidden_size: c_int = 2048,
    intermediate_size: c_int = 8192,
    num_attention_heads: c_int = 32,
    num_key_value_heads: ?c_int = 8,
    max_position_embeddings: c_int = 131072,
    rms_norm_eps: f32 = 1e-5,
    rope_theta: f32 = 500000.0,
    mlp_bias: bool = false,
    attention_bias: bool = false,
    tie_word_embeddings: bool = true,
    vocab_size: c_int = 128256,
    num_hidden_layers: c_int = 16,
    quantization: ?mlx.QuantConfig = null,
    rope_scaling: RopeScalingConfig,
    torch_dtype: []u8,
    pub const RopeScalingConfig = struct {
        factor: f32 = 32.0,
        high_freq_factor: f32 = 4.0,
        low_freq_factor: f32 = 1.0,
        original_max_position_embeddings: f32 = 8192,
    };
};
