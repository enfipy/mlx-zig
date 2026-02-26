//! qwen.zig - Qwen 2.5 (QwQ, R1, R1-Math, Coder, Olympic)
//!
//! Copyright 2025

const std = @import("std");
const mlx = @import("mlx.zig");
const utils = @import("utils.zig");

pub const MLP = struct {
    const Self = @This();
    base: mlx.Module = undefined,
    gate_proj: *mlx.Linear = undefined,
    down_proj: *mlx.Linear = undefined,
    up_proj: *mlx.Linear = undefined,

    pub fn init(key: []const u8, model_config: ModelConfig, mlx_config: *mlx.MLXConfig) !*Self {
        const self = try mlx_config.allocator.create(Self);
        self.* = .{ .base = mlx.Module.init(mlx_config.allocator, mlx_config.stream) };
        self.gate_proj = try mlx.Linear.init(
            try self.base.alloc_join(key, "gate_proj"),
            false,
            model_config.quantization,
            mlx_config,
        );
        self.down_proj = try mlx.Linear.init(
            try self.base.alloc_join(key, "down_proj"),
            false,
            model_config.quantization,
            mlx_config,
        );
        self.up_proj = try mlx.Linear.init(
            try self.base.alloc_join(key, "up_proj"),
            false,
            model_config.quantization,
            mlx_config,
        );
        return self;
    }

    pub fn forward(self: *Self, result: *mlx.Array, x: mlx.Array) !void {
        var gate = mlx.array_new();
        var up = mlx.array_new();
        defer {
            mlx.array_free(gate);
            mlx.array_free(up);
        }
        try self.gate_proj.forward(&gate, x);
        try self.up_proj.forward(&up, x);
        try mlx.silu(&gate, gate, self.base.stream);
        try mlx.multiply(&gate, gate, up, self.base.stream);
        try self.down_proj.forward(result, gate);
    }

    pub fn deinit(self: *Self) void {
        self.gate_proj.deinit();
        self.down_proj.deinit();
        self.up_proj.deinit();
        self.base.deinit();
        self.base.allocator.destroy(self);
    }
};

pub const Attention = struct {
    const Self = @This();
    base: mlx.Module = undefined,
    n_heads: c_int = undefined,
    n_kv_heads: c_int = undefined,
    head_dim: c_int = undefined,
    scale: f32 = undefined,
    q_proj: *mlx.Linear = undefined,
    k_proj: *mlx.Linear = undefined,
    v_proj: *mlx.Linear = undefined,
    o_proj: *mlx.Linear = undefined,
    rope: *mlx.RoPE = undefined,

    pub fn init(key: []const u8, model_config: ModelConfig, mlx_config: *mlx.MLXConfig) !*Self {
        const self = try mlx_config.allocator.create(Self);
        self.* = .{
            .base = mlx.Module.init(mlx_config.allocator, mlx_config.stream),
            .q_proj = undefined,
            .k_proj = undefined,
            .v_proj = undefined,
            .o_proj = undefined,
            .rope = undefined,
        };
        self.n_heads = model_config.num_attention_heads;
        self.n_kv_heads = model_config.num_key_value_heads orelse model_config.num_attention_heads;
        self.head_dim = @divExact(model_config.hidden_size, model_config.num_attention_heads);
        self.scale = 1.0 / @sqrt(@as(f32, @floatFromInt(self.head_dim)));
        self.q_proj = try mlx.Linear.init(
            try self.base.alloc_join(key, "q_proj"),
            true,
            model_config.quantization,
            mlx_config,
        );
        self.k_proj = try mlx.Linear.init(
            try self.base.alloc_join(key, "k_proj"),
            true,
            model_config.quantization,
            mlx_config,
        );
        self.v_proj = try mlx.Linear.init(
            try self.base.alloc_join(key, "v_proj"),
            true,
            model_config.quantization,
            mlx_config,
        );
        self.o_proj = try mlx.Linear.init(
            try self.base.alloc_join(key, "o_proj"),
            false,
            model_config.quantization,
            mlx_config,
        );
        self.rope = try mlx.RoPE.init(
            self.head_dim,
            false,
            model_config.rope_theta,
            1.0,
            mlx_config,
        );
        return self;
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
        defer {
            mlx.array_free(q);
            mlx.array_free(k);
            mlx.array_free(v);
        }
        try self.q_proj.forward(&q, x);
        try self.k_proj.forward(&k, x);
        try self.v_proj.forward(&v, x);
        try mlx.reshape_to_heads(&q, q, self.n_heads, self.head_dim, self.base.stream);
        try mlx.reshape_to_heads(&k, k, self.n_kv_heads, self.head_dim, self.base.stream);
        try mlx.reshape_to_heads(&v, v, self.n_kv_heads, self.head_dim, self.base.stream);
        try self.rope.forward(&q, q, offset);
        try self.rope.forward(&k, k, offset);
        if (cache) |c| try c.update(&k, &v, null, self.base.stream);
        const mask_array = mask orelse mlx.C.mlx_array_empty;
        const memory_threshold = mlx.C.mlx_optional_int{
            .has_value = false,
            .value = 0,
        };
        try mlx.op_status(
            mlx.C.mlx_fast_scaled_dot_product_attention(
                result,
                q,
                k,
                v,
                self.scale,
                mask_array,
                memory_threshold,
                self.base.stream,
            ),
        );
        try mlx.reshape_from_heads(result, result.*, self.base.stream);
        try self.o_proj.forward(result, result.*);
    }

    pub fn deinit(self: *Self) void {
        self.q_proj.deinit();
        self.k_proj.deinit();
        self.v_proj.deinit();
        self.o_proj.deinit();
        self.rope.deinit();
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
        self.mlp = try MLP.init(try self.base.alloc_join(key, "mlp"), model_config, mlx_config);
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

    pub fn forward(
        self: *Self,
        result: *mlx.Array,
        x: mlx.Array,
        mask: ?mlx.Array,
        cache: ?*mlx.KVCache,
        offset: c_int,
    ) !void {
        var attn_input = mlx.array_new();
        var attn_output = mlx.array_new();
        var mlp_input = mlx.array_new();
        var mlp_output = mlx.array_new();
        defer {
            mlx.array_free(attn_input);
            mlx.array_free(attn_output);
            mlx.array_free(mlp_input);
            mlx.array_free(mlp_output);
        }
        try self.input_layernorm.forward(&attn_input, x);
        try self.self_attn.forward(&attn_output, attn_input, mask, cache, offset);
        try mlx.add(&attn_output, x, attn_output, self.base.stream);
        try self.post_attention_layernorm.forward(&mlp_input, attn_output);
        try self.mlp.forward(&mlp_output, mlp_input);
        try mlx.add(result, attn_output, mlp_output, self.base.stream);
    }

    pub fn deinit(self: *Self) void {
        self.self_attn.deinit();
        self.mlp.deinit();
        self.input_layernorm.deinit();
        self.post_attention_layernorm.deinit();
        self.base.deinit();
        self.base.allocator.destroy(self);
    }
};

pub const ModelConfig = @import("mlx.zig").ModelConfig;
pub const Model = mlx.Model(TransformerBlock, ModelConfig);
pub const Transformer = mlx.Transformer(Model, ModelConfig);

test "qwen.zig - qwen2.5-coder" {
    std.debug.print("\n=== QWEN.ZIG - QWEN2.5-CODER ===\n\n", .{});
    const allocator = std.testing.allocator;
    const initial_tokens = [_]u32{ 151659, 750, 3974 };
    const num_tokens_to_generate = 10;
    var transformer = try Transformer.init(allocator, "Qwen2.5-Coder-1.5B-4bit");
    defer transformer.deinit();
    const generated_tokens = try transformer.generate(&initial_tokens, num_tokens_to_generate);
    defer allocator.free(generated_tokens);
    std.debug.print("\nGenerated sequence: ", .{});
    for (generated_tokens) |token| {
        std.debug.print("{d} ", .{token});
    }
    std.debug.print("\n", .{});
}

test "qwen.zig - olympic-coder" {
    std.debug.print("\n=== QWEN.ZIG - OLYMPIC-CODER ===\n\n", .{});
    const allocator = std.testing.allocator;
    const initial_tokens = [_]u32{ 151644, 872, 198, 7985 };
    const num_tokens_to_generate = 10;
    var transformer = try Transformer.init(allocator, "OlympicCoder-7B-4bit");
    defer transformer.deinit();
    const generated_tokens = try transformer.generate(&initial_tokens, num_tokens_to_generate);
    defer allocator.free(generated_tokens);
    std.debug.print("\nGenerated sequence: ", .{});
    for (generated_tokens) |token| {
        std.debug.print("{d} ", .{token});
    }
    std.debug.print("\n", .{});
}
