const std = @import("std");
const core = @import("mlx_core.zig");

const C = core.C;
const Array = core.Array;
const Stream = core.Stream;
const OptionalFloat = core.OptionalFloat;
const MLXError = core.MLXError;

const stream_free = core.stream_free;
const array_new = core.array_new;
const array_free = core.array_free;
const array_set = core.array_set;
const array_eval = core.array_eval;
const op_status = core.op_status;
const einsum = core.einsum;
const add = core.add;
const take = core.take;
const fast_rms_norm = core.fast_rms_norm;
const fast_layer_norm = core.fast_layer_norm;
const fast_rope = core.fast_rope;
const astype = core.astype;

pub const QuantConfig = struct {
    group_size: c_int,
    bits: c_int,
};

pub const MLXConfig = struct {
    allocator: std.mem.Allocator,
    stream: Stream,
    weights_hash: std.StringHashMap(*Array),
    dtype: C.mlx_dtype,

    pub fn init(allocator: std.mem.Allocator, mlx_dtype: C.mlx_dtype) !MLXConfig {
        return MLXConfig{
            .allocator = allocator,
            .stream = C.mlx_default_gpu_stream_new(),
            .weights_hash = std.StringHashMap(*Array).init(allocator),
            .dtype = mlx_dtype,
        };
    }

    pub fn deinit(self: *@This()) void {
        stream_free(self.stream);
        self.weights_hash.deinit();
    }
};

pub const Module = struct {
    const Self = @This();
    allocator: std.mem.Allocator,
    stream: Stream,
    allocs_to_free: std.array_list.Managed([]const u8),

    pub fn init(allocator: std.mem.Allocator, stream: Stream) Self {
        return .{
            .allocator = allocator,
            .stream = stream,
            .allocs_to_free = std.array_list.Managed([]const u8).init(allocator),
        };
    }

    pub fn alloc_dupe(self: *Self, key: []const u8) ![]const u8 {
        const owned_key = try self.allocator.dupe(u8, key);
        try self.allocs_to_free.append(owned_key);
        return owned_key;
    }

    pub fn alloc_join(self: *Self, parent: []const u8, name: anytype) ![]const u8 {
        const name_info = @typeInfo(@TypeOf(name));
        const is_null_name = @TypeOf(name) == @TypeOf(null);
        const is_empty_pointer_name = name_info == .pointer and name.len == 0;
        const is_integer_name = name_info == .int or name_info == .comptime_int;

        var owned_key: []const u8 = undefined;
        if (is_null_name) {
            owned_key = try self.allocator.dupe(u8, parent);
        } else if (is_empty_pointer_name) {
            owned_key = try self.allocator.dupe(u8, parent);
        } else if (is_integer_name) {
            owned_key = try std.fmt.allocPrint(self.allocator, "{s}.{d}", .{ parent, name });
        } else {
            owned_key = try std.fmt.allocPrint(self.allocator, "{s}.{s}", .{ parent, name });
        }
        try self.allocs_to_free.append(owned_key);
        return owned_key;
    }

    pub fn deinit(self: *Self) void {
        for (self.allocs_to_free.items) |key| self.allocator.free(key);
        self.allocs_to_free.deinit();
    }
};

pub const Weight = struct {
    const Self = @This();
    base: Module,
    weight: Array,
    is_quantized: bool,
    scales: ?Array,
    biases: ?Array,
    group_size: ?c_int,
    bits: ?c_int,

    pub fn init(key: []const u8, quant_config: ?QuantConfig, mlx_config: *MLXConfig) !*Self {
        const is_quantized = quant_config != null;
        var scales_value: ?Array = null;
        var biases_value: ?Array = null;
        var group_size_value: ?c_int = null;
        var bits_value: ?c_int = null;
        if (is_quantized) {
            scales_value = array_new();
            biases_value = array_new();
            group_size_value = quant_config.?.group_size;
            bits_value = quant_config.?.bits;
        }
        var self = try mlx_config.allocator.create(Self);
        self.* = .{
            .base = Module.init(mlx_config.allocator, mlx_config.stream),
            .weight = array_new(),
            .is_quantized = is_quantized,
            .scales = scales_value,
            .biases = biases_value,
            .group_size = group_size_value,
            .bits = bits_value,
        };
        const weight_key = try self.base.alloc_join(key, "weight");
        try mlx_config.weights_hash.put(weight_key, &self.weight);
        if (is_quantized) {
            const scales_key = try self.base.alloc_join(key, "scales");
            try mlx_config.weights_hash.put(scales_key, &self.scales.?);

            const biases_key = try self.base.alloc_join(key, "biases");
            try mlx_config.weights_hash.put(biases_key, &self.biases.?);
        }
        return self;
    }

    pub fn deinit(self: *Self) void {
        array_free(self.weight);
        if (self.is_quantized) {
            array_free(self.scales.?);
            array_free(self.biases.?);
        }
        self.base.deinit();
        self.base.allocator.destroy(self);
    }

    pub fn forward(self: *Self, result: *Array, x: Array) MLXError!void {
        if (self.is_quantized) {
            try op_status(
                C.mlx_quantized_matmul(
                    result,
                    x,
                    self.weight,
                    self.scales.?,
                    self.biases.?,
                    true,
                    self.group_size.?,
                    self.bits.?,
                    self.base.stream,
                ),
            );
        } else {
            try einsum(result, &[_]Array{ x, self.weight }, "blh,dh->bld", self.base.stream);
        }
    }

    pub fn dequantize(self: *Self) MLXError!void {
        if (self.is_quantized) {
            var temp = array_new();
            defer array_free(temp);
            try op_status(
                C.mlx_dequantize(
                    &temp,
                    self.weight,
                    self.scales.?,
                    self.biases.?,
                    self.group_size.?,
                    self.bits.?,
                    self.base.stream,
                ),
            );
            try array_set(&self.weight, temp);
            try array_eval(self.weight);
            array_free(self.scales.?);
            array_free(self.biases.?);
            self.scales = null;
            self.biases = null;
            self.is_quantized = false;
        }
    }
};

pub const Linear = struct {
    const Self = @This();
    base: Module,
    weight: *Weight,
    bias: ?Array,
    has_bias: bool,
    is_sanitized: bool,

    pub fn init(
        key: []const u8,
        has_bias: bool,
        quant_config: ?QuantConfig,
        mlx_config: *MLXConfig,
    ) !*Self {
        const self = try mlx_config.allocator.create(Self);
        const is_sanitized = quant_config != null;
        var bias_value: ?Array = null;
        if (has_bias) {
            bias_value = array_new();
        }
        self.* = .{
            .base = Module.init(mlx_config.allocator, mlx_config.stream),
            .weight = try Weight.init(key, quant_config, mlx_config),
            .is_sanitized = is_sanitized,
            .has_bias = has_bias,
            .bias = bias_value,
        };
        if (has_bias) {
            const bias_key = try self.base.alloc_join(key, "bias");
            try mlx_config.weights_hash.put(bias_key, &self.bias.?);
        }
        return self;
    }

    pub fn sanitize(self: *Self) !void {
        if (self.is_sanitized) {
            return;
        }
        try op_status(
            C.mlx_swapaxes(
                &self.weight.weight,
                self.weight.weight,
                0,
                1,
                self.base.stream,
            ),
        );
        try array_eval(self.weight.weight);
        self.is_sanitized = true;
    }
    pub fn deinit(self: *Self) void {
        self.weight.deinit();
        if (self.has_bias) {
            array_free(self.bias.?);
        }
        self.base.deinit();
        self.base.allocator.destroy(self);
    }

    pub fn forward(self: *Self, result: *Array, x: Array) MLXError!void {
        if (self.weight.is_quantized) {
            try self.weight.forward(result, x);
            if (self.has_bias) {
                try add(result, result.*, self.bias.?, self.base.stream);
            }
        } else {
            try self.sanitize();
            if (self.has_bias) {
                try op_status(
                    C.mlx_addmm(
                        result,
                        self.bias.?,
                        x,
                        self.weight.weight,
                        1.0,
                        1.0,
                        self.base.stream,
                    ),
                );
            } else {
                try op_status(C.mlx_matmul(result, x, self.weight.weight, self.base.stream));
            }
        }
    }
};

pub const Embedding = struct {
    const Self = @This();
    base: Module,
    weight: *Weight,
    is_sanitized: bool = false,

    pub fn init(key: []const u8, quant_config: ?QuantConfig, mlx_config: *MLXConfig) !*Self {
        const self = try mlx_config.allocator.create(Self);
        self.* = .{
            .base = Module.init(mlx_config.allocator, mlx_config.stream),
            .weight = try Weight.init(key, quant_config, mlx_config),
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.weight.deinit();
        self.base.deinit();
        self.base.allocator.destroy(self);
    }

    pub fn sanitize(self: *Self) MLXError!void {
        if (self.is_sanitized) {
            return;
        }
        try self.weight.dequantize();
        self.is_sanitized = true;
    }

    pub fn forward(self: *Self, result: *Array, toks: Array) MLXError!void {
        try self.sanitize();
        try take(result, self.weight.weight, toks, 0, self.base.stream);
    }

    pub fn as_linear(self: *Self, result: *Array, x: Array) MLXError!void {
        try self.weight.forward(result, x);
    }
};

pub const RMSNorm = struct {
    const Self = @This();
    base: Module,
    eps: f32,
    weight: Array,

    pub fn init(key: []const u8, eps: f32, mlx_config: *MLXConfig) !*Self {
        const self = try mlx_config.allocator.create(Self);
        self.* = .{
            .base = Module.init(mlx_config.allocator, mlx_config.stream),
            .weight = array_new(),
            .eps = eps,
        };
        const weight_key = try self.base.alloc_join(key, "weight");
        try mlx_config.weights_hash.put(weight_key, &self.weight);
        return self;
    }

    pub fn deinit(self: *Self) void {
        array_free(self.weight);
        self.base.deinit();
        self.base.allocator.destroy(self);
    }

    pub fn forward(self: *Self, result: *Array, x: Array) MLXError!void {
        try fast_rms_norm(result, x, self.weight, self.eps, self.base.stream);
    }
};

pub const LayerNorm = struct {
    const Self = @This();
    base: Module,
    eps: f32,
    weight: Array,
    bias: Array,

    pub fn init(key: []const u8, eps: f32, mlx_config: *MLXConfig) !*Self {
        const self = try mlx_config.allocator.create(Self);
        self.* = .{
            .base = Module.init(mlx_config.allocator, mlx_config.stream),
            .eps = eps,
            .weight = array_new(),
            .bias = array_new(),
        };
        const weight_key = try self.base.alloc_join(key, "weight");
        try mlx_config.weights_hash.put(weight_key, &self.weight);
        const bias_key = try self.base.alloc_join(key, "bias");
        try mlx_config.weights_hash.put(bias_key, &self.bias);
        return self;
    }

    pub fn deinit(self: *Self) void {
        array_free(self.weight);
        array_free(self.bias);
        self.base.deinit();
        self.base.allocator.destroy(self);
    }

    pub fn forward(self: *Self, result: *Array, x: Array) MLXError!void {
        try fast_layer_norm(result, x, self.weight, self.bias, self.eps, self.base.stream);
    }
};

pub const Conv1d = struct {
    const Self = @This();
    base: Module,
    stride: c_int,
    padding: c_int,
    dilation: c_int,
    groups: c_int,
    has_bias: bool,
    weight: Array,
    is_sanitized: bool = false,
    bias: ?Array,

    pub fn init(
        key: []const u8,
        stride: c_int,
        padding: c_int,
        dilation: c_int,
        groups: c_int,
        has_bias: bool,
        mlx_config: *MLXConfig,
    ) !*Self {
        const self = try mlx_config.allocator.create(Self);
        var bias_value: ?Array = null;
        if (has_bias) {
            bias_value = array_new();
        }
        self.* = .{
            .base = Module.init(mlx_config.allocator, mlx_config.stream),
            .stride = stride,
            .padding = padding,
            .dilation = dilation,
            .groups = groups,
            .has_bias = has_bias,
            .weight = array_new(),
            .bias = bias_value,
        };
        const weight_key = try self.base.alloc_join(key, "weight");
        try mlx_config.weights_hash.put(weight_key, &self.weight);
        if (has_bias) {
            const bias_key = try self.base.alloc_join(key, "bias");
            try mlx_config.weights_hash.put(bias_key, &self.bias.?);
        }
        return self;
    }

    pub fn deinit(self: *Self) void {
        array_free(self.weight);
        if (self.has_bias) {
            array_free(self.bias.?);
        }
        self.base.deinit();
        self.base.allocator.destroy(self);
    }

    pub fn sanitize(self: *Self) !void {
        if (self.is_sanitized) {
            return;
        }
        try op_status(C.mlx_swapaxes(&self.weight, self.weight, 1, 2, self.base.stream));
        try array_eval(self.weight);
        self.is_sanitized = true;
    }

    pub fn forward(self: *Self, result: *Array, x: Array) !void {
        try self.sanitize();
        try op_status(
            C.mlx_conv1d(
                result,
                x,
                self.weight,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
                self.base.stream,
            ),
        );
        if (self.has_bias) {
            try add(result, result.*, self.bias.?, self.base.stream);
        }
    }
};

pub const RoPE = struct {
    const Self = @This();
    base: Module,
    dims: c_int,
    traditional: bool,
    theta_base: OptionalFloat,
    scale: f32,
    freqs: Array,
    dtype: C.mlx_dtype,

    pub fn init(
        dims: c_int,
        traditional: bool,
        theta_base: f32,
        scale: f32,
        mlx_config: *MLXConfig,
    ) !*Self {
        const self = try mlx_config.allocator.create(Self);
        self.* = .{
            .base = Module.init(mlx_config.allocator, mlx_config.stream),
            .dims = dims,
            .traditional = traditional,
            .theta_base = OptionalFloat{ .has_value = true, .value = theta_base },
            .scale = scale,
            .freqs = C.mlx_array_empty,
            .dtype = mlx_config.dtype,
        };
        return self;
    }

    pub fn forward(self: *Self, result: *Array, x: Array, offset: c_int) !void {
        try fast_rope(
            result,
            x,
            self.dims,
            self.traditional,
            self.theta_base,
            self.scale,
            offset,
            self.freqs,
            self.base.stream,
        );
        try astype(result, result.*, self.dtype, self.base.stream);
    }

    pub fn deinit(self: *Self) void {
        self.base.deinit();
        self.base.allocator.destroy(self);
    }
};
