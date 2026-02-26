//! mlx.zig - MLX Bindings
//!
//! Copyright 2025 Joe

const std = @import("std");
pub const C = @cImport({
    @cInclude("mlx/c/mlx.h");
    @cInclude("stdio.h");
});

fn default_io() std.Io {
    return std.Io.Threaded.global_single_threaded.io();
}

inline fn discard(_: anytype) void {}

pub fn now_ms() i128 {
    var ts: std.posix.timespec = undefined;
    const err = std.posix.errno(std.posix.system.clock_gettime(.MONOTONIC, &ts));
    if (err == .SUCCESS) {
        return @as(i128, ts.sec) * std.time.ms_per_s + @divTrunc(ts.nsec, std.time.ns_per_ms);
    }
    return 0;
}

const MAX_SAFE_TENSORS_ENTRIES: usize = 131072;

inline fn require_or_fail(condition: bool, err: MLXError) MLXError!void {
    if (condition) {
        std.debug.assert(condition);
        return;
    }
    return err;
}

/// ============================================================================
/// Types & Constants
/// ============================================================================
pub const BOOL: C.mlx_dtype = C.MLX_BOOL;
pub const INT32: C.mlx_dtype = C.MLX_INT32;
pub const UINT32: C.mlx_dtype = C.MLX_UINT32;
pub const FLOAT16: C.mlx_dtype = C.MLX_FLOAT16;
pub const FLOAT32: C.mlx_dtype = C.MLX_FLOAT32;
pub const FLOAT64: C.mlx_dtype = C.MLX_FLOAT64;
pub const BFLOAT16: C.mlx_dtype = C.MLX_BFLOAT16;
pub const Array = C.mlx_array;
pub const String = C.mlx_string;
pub const Stream = C.mlx_stream;
pub const VectorArray = C.mlx_vector_array;
pub const MapStrArr = C.mlx_map_string_to_array;
pub const MapStrStr = C.mlx_map_string_to_string;
pub const OptionalFloat = C.mlx_optional_float;

/// ============================================================================
/// Scalar Operations
/// ============================================================================
pub const add = define_binary_op("mlx_add");
pub const subtract = define_binary_op("mlx_subtract");
pub const multiply = define_binary_op("mlx_multiply");
pub const divide = define_binary_op("mlx_divide");
pub const power = define_binary_op("mlx_power");
pub const greater = define_binary_op("mlx_greater");
pub const greater_equal = define_binary_op("mlx_greater_equal");
pub const less = define_binary_op("mlx_less");
pub const less_equal = define_binary_op("mlx_less_equal");
pub const logical_or = define_binary_op("mlx_logical_or");
pub const logical_and = define_binary_op("mlx_logical_and");
pub const matmul = define_binary_op("mlx_matmul");
pub const minimum = define_binary_op("mlx_minimum");
pub const maximum = define_binary_op("mlx_maximum");
pub const logical_not = define_unary_op("mlx_logical_not");
pub const isnan = define_unary_op("mlx_isnan");
pub const sigmoid = define_unary_op("mlx_sigmoid");
pub const sin = define_unary_op("mlx_sin");
pub const cos = define_unary_op("mlx_cos");
pub const exp = define_unary_op("mlx_exp");
pub const abs = define_unary_op("mlx_abs");
pub const square = define_unary_op("mlx_square");
pub const log = define_unary_op("mlx_log");
pub const log10 = define_unary_op("mlx_log10");

fn item_c_func_name_int32(comptime child: type) ?[]const u8 {
    if (child == u32) return "mlx_array_item_uint32";
    if (child == c_uint) return "mlx_array_item_uint32";
    if (child == i32) return "mlx_array_item_int32";
    if (child == c_int) return "mlx_array_item_int32";
    return null;
}

fn item_c_func_name_int64(comptime child: type) ?[]const u8 {
    if (child == u64) return "mlx_array_item_uint64";
    if (child == i64) return "mlx_array_item_int64";
    return null;
}

fn item_c_func_name_int(comptime child: type) ?[]const u8 {
    if (item_c_func_name_int32(child)) |name| return name;
    return item_c_func_name_int64(child);
}

fn item_c_func_name_small_int(comptime child: type) ?[]const u8 {
    if (child == u8) return "mlx_array_item_uint8";
    if (child == i8) return "mlx_array_item_int8";
    if (child == u16) return "mlx_array_item_uint16";
    if (child == i16) return "mlx_array_item_int16";
    return null;
}

fn item_c_func_name_scalar(comptime child: type) ?[]const u8 {
    if (child == f32) return "mlx_array_item_float32";
    if (child == f64) return "mlx_array_item_float64";
    if (child == bool) return "mlx_array_item_bool";
    return null;
}

fn item_c_func_name(comptime child: type) []const u8 {
    if (item_c_func_name_int(child)) |name| return name;
    if (item_c_func_name_small_int(child)) |name| return name;
    if (item_c_func_name_scalar(child)) |name| return name;
    @compileError("Unsupported item type: " ++ @typeName(child));
}

pub fn item(dest: anytype, arr: Array) MLXError!void {
    if (arr.ctx == null) return MLXError.InvalidArray;
    if (@intFromPtr(dest) == 0) return MLXError.InvalidArray;
    std.debug.assert(@intFromPtr(dest) > 0);
    std.debug.assert(arr.ctx != null);
    const T = @TypeOf(dest);
    const info = @typeInfo(T);
    if (info == .pointer) {
        std.debug.assert(info == .pointer);
    } else {
        @compileError("Expected pointer, got " ++ @typeName(T));
    }
    const child = info.pointer.child;
    const c_func_name = comptime item_c_func_name(child);
    std.debug.assert(c_func_name.len > 0);
    try op_call_checked(@field(C, c_func_name)(dest, arr), c_func_name);
}

pub fn astype(result: *Array, x: anytype, dtype: C.mlx_dtype, stream: Stream) MLXError!void {
    const x_conv = array_converter(x);
    defer x_conv.deinit();
    try op_call_checked(C.mlx_astype(result, x_conv.arr, dtype, stream), "mlx_astype");
}

/// ============================================================================
/// Array Operations
/// ============================================================================
pub const all_max = define_reduce_all_op("mlx_max_all");
pub const all_min = define_reduce_all_op("mlx_min_all");
pub const max = define_reduce_axes_op("mlx_max");
pub const min = define_reduce_axes_op("mlx_min");
pub const rfft = define_fft_op("mlx_fft_rfft");
pub const irfft = define_fft_op("mlx_fft_irfft");
pub const fft = define_fft_op("mlx_fft_fft");
pub const ifft = define_fft_op("mlx_fft_ifft");
pub const fft2 = define_fftn_op("mlx_fft_fft2");
pub const ifft2 = define_fftn_op("mlx_fft_ifft2");
pub const rfft2 = define_fftn_op("mlx_fft_rfft2");
pub const irfft2 = define_fftn_op("mlx_fft_irfft2");
pub const fftn = define_fftn_op("mlx_fft_fftn");
pub const ifftn = define_fftn_op("mlx_fft_ifftn");
pub const rfftn = define_fftn_op("mlx_fft_rfftn");
pub const irfftn = define_fftn_op("mlx_fft_irfftn");
pub const array_new = C.mlx_array_new;
pub const array_new_float = C.mlx_array_new_float;
pub const array_dim = C.mlx_array_dim;
pub const array_shape = C.mlx_array_shape;

pub fn array_new_data(
    data: *const anyopaque,
    shape: []const c_int,
    dtype: C.mlx_dtype,
) MLXError!Array {
    if (shape.len == 0) return MLXError.InvalidArray;
    std.debug.assert(dtype >= 0);
    std.debug.assert(shape.len > 0);
    const arr = C.mlx_array_new_data(data, shape.ptr, @intCast(shape.len), dtype);
    if (arr.ctx == null) return MLXError.InvalidArray;
    std.debug.assert(arr.ctx != null);
    return arr;
}

pub fn array_set_data(
    arr: *Array,
    data: *const anyopaque,
    shape: []const c_int,
    dtype: C.mlx_dtype,
) MLXError!void {
    if (shape.len == 0) return MLXError.InvalidArray;
    std.debug.assert(shape.len > 0);
    try op_call_checked(
        C.mlx_array_set_data(arr, data, shape.ptr, @intCast(shape.len), dtype),
        "mlx_array_set_data",
    );
}

pub fn where(result: *Array, cond: Array, x: anytype, y: anytype, stream: Stream) MLXError!void {
    const x_conv = array_converter(x);
    const y_conv = array_converter(y);
    defer {
        x_conv.deinit();
        y_conv.deinit();
    }
    try op_call_checked(C.mlx_where(result, cond, x_conv.arr, y_conv.arr, stream), "mlx_where");
}

pub fn take(result: *Array, x: Array, indices: anytype, axis: c_int, stream: Stream) MLXError!void {
    const indices_conv = array_converter(indices);
    defer indices_conv.deinit();
    try op_call_checked(C.mlx_take(result, x, indices_conv.arr, axis, stream), "mlx_take");
}

pub fn pad(
    result: *Array,
    x: Array,
    axes: []const c_int,
    low_pad: []const c_int,
    high_pad: []const c_int,
    pad_value: anytype,
    pad_mode: [*:0]const u8,
    stream: Stream,
) MLXError!void {
    if (x.ctx == null) return MLXError.InvalidArray;
    if (axes.len == 0) return MLXError.InvalidArray;
    if (axes.len != low_pad.len) return MLXError.InvalidArray;
    if (axes.len != high_pad.len) return MLXError.InvalidArray;
    std.debug.assert(x.ctx != null);
    std.debug.assert(axes.len > 0);
    std.debug.assert(axes.len == low_pad.len);
    std.debug.assert(axes.len == high_pad.len);
    const pad_val_conv = array_converter(pad_value);
    defer pad_val_conv.deinit();
    if (pad_val_conv.arr.ctx == null) return MLXError.InvalidArray;
    std.debug.assert(pad_val_conv.arr.ctx != null);
    try op_call_checked(
        C.mlx_pad(
            result,
            x,
            axes.ptr,
            axes.len,
            low_pad.ptr,
            low_pad.len,
            high_pad.ptr,
            high_pad.len,
            pad_val_conv.arr,
            pad_mode,
            stream,
        ),
        "mlx_pad",
    );
}

pub fn slice(
    result: *Array,
    x: Array,
    start: []const c_int,
    stop: []const c_int,
    strides: []const c_int,
    stream: Stream,
) MLXError!void {
    if (x.ctx == null) return MLXError.InvalidArray;
    if (start.len == 0) return MLXError.InvalidArray;
    if (start.len != stop.len) return MLXError.InvalidArray;
    if (start.len != strides.len) return MLXError.InvalidArray;
    std.debug.assert(x.ctx != null);
    std.debug.assert(start.len > 0);
    std.debug.assert(start.len == stop.len);
    std.debug.assert(start.len == strides.len);
    try op_call_checked(
        C.mlx_slice(
            result,
            x,
            start.ptr,
            start.len,
            stop.ptr,
            stop.len,
            strides.ptr,
            strides.len,
            stream,
        ),
        "mlx_slice",
    );
}

pub fn as_strided(
    result: *Array,
    x: Array,
    shape: []const c_int,
    strides: []const i64,
    offset: u64,
    stream: Stream,
) MLXError!void {
    if (x.ctx == null) return MLXError.InvalidArray;
    if (shape.len == 0) return MLXError.InvalidArray;
    if (shape.len != strides.len) return MLXError.InvalidArray;
    std.debug.assert(x.ctx != null);
    std.debug.assert(shape.len > 0);
    std.debug.assert(shape.len == strides.len);
    try op_call_checked(
        C.mlx_as_strided(
            result,
            x,
            shape.ptr,
            shape.len,
            strides.ptr,
            strides.len,
            offset,
            stream,
        ),
        "mlx_as_strided",
    );
}

pub fn expand_dims(result: *Array, x: Array, axes: []const c_int, stream: Stream) MLXError!void {
    if (x.ctx == null) return MLXError.InvalidArray;
    if (axes.len == 0) return MLXError.InvalidArray;
    std.debug.assert(x.ctx != null);
    std.debug.assert(axes.len > 0);
    try op_call_checked(
        C.mlx_expand_dims(result, x, axes.ptr, axes.len, stream),
        "mlx_expand_dims",
    );
}

pub fn reshape(result: *Array, x: Array, shape: []const c_int, stream: Stream) MLXError!void {
    if (x.ctx == null) return MLXError.InvalidArray;
    if (shape.len == 0) return MLXError.InvalidArray;
    std.debug.assert(x.ctx != null);
    std.debug.assert(shape.len > 0);
    try op_call_checked(C.mlx_reshape(result, x, shape.ptr, shape.len, stream), "mlx_reshape");
}

pub fn softmax(
    result: *Array,
    x: Array,
    axes: []const c_int,
    precise: bool,
    stream: Stream,
) MLXError!void {
    if (x.ctx == null) return MLXError.InvalidArray;
    if (axes.len == 0) return MLXError.InvalidArray;
    std.debug.assert(x.ctx != null);
    std.debug.assert(axes.len > 0);
    try op_call_checked(
        C.mlx_softmax(result, x, axes.ptr, axes.len, precise, stream),
        "mlx_softmax",
    );
}

pub fn ones(
    result: *Array,
    shape: []const c_int,
    dtype: C.mlx_dtype,
    stream: Stream,
) MLXError!void {
    if (shape.len == 0) return MLXError.InvalidArray;
    if (dtype < 0) return MLXError.InvalidArray;
    std.debug.assert(shape.len > 0);
    std.debug.assert(dtype >= 0);
    try op_call_checked(C.mlx_ones(result, shape.ptr, shape.len, dtype, stream), "mlx_ones");
}

pub fn zeros(
    result: *Array,
    shape: []const c_int,
    dtype: C.mlx_dtype,
    stream: Stream,
) MLXError!void {
    if (shape.len == 0) return MLXError.InvalidArray;
    if (dtype < 0) return MLXError.InvalidArray;
    std.debug.assert(shape.len > 0);
    std.debug.assert(dtype >= 0);
    try op_call_checked(C.mlx_zeros(result, shape.ptr, shape.len, dtype, stream), "mlx_zeros");
}

pub fn argmax(result: *Array, x: Array, axis: c_int, keepdims: bool, stream: Stream) MLXError!void {
    try op_call_checked(C.mlx_argmax(result, x, axis, keepdims, stream), "mlx_argmax");
}

pub fn argsort(result: *Array, x: Array, axis: c_int, stream: Stream) MLXError!void {
    if (x.ctx == null) return MLXError.InvalidArray;
    std.debug.assert(x.ctx != null);
    try op_call_checked(C.mlx_argsort(result, x, axis, stream), "mlx_argsort");
}

pub fn cumsum(
    result: *Array,
    x: Array,
    axis: c_int,
    reverse: bool,
    inclusive: bool,
    stream: Stream,
) MLXError!void {
    if (x.ctx == null) return MLXError.InvalidArray;
    std.debug.assert(x.ctx != null);
    try op_call_checked(
        C.mlx_cumsum(result, x, axis, reverse, inclusive, stream),
        "mlx_cumsum",
    );
}

pub fn take_along_axis(
    result: *Array,
    x: Array,
    indices: Array,
    axis: c_int,
    stream: Stream,
) MLXError!void {
    if (x.ctx == null or indices.ctx == null) return MLXError.InvalidArray;
    std.debug.assert(x.ctx != null);
    std.debug.assert(indices.ctx != null);
    try op_call_checked(
        C.mlx_take_along_axis(result, x, indices, axis, stream),
        "mlx_take_along_axis",
    );
}

pub fn topk(result: *Array, x: Array, k: c_int, axis: c_int, stream: Stream) MLXError!void {
    if (x.ctx == null) return MLXError.InvalidArray;
    if (k <= 0) return MLXError.InvalidArray;
    std.debug.assert(x.ctx != null);
    std.debug.assert(k > 0);
    try op_call_checked(C.mlx_topk(result, x, k, axis, stream), "mlx_topk");
}

pub fn random_categorical(
    result: *Array,
    logits: Array,
    axis: c_int,
    stream: Stream,
) MLXError!void {
    if (@intFromPtr(result) == 0) {
        return MLXError.InvalidArray;
    }
    if (logits.ctx == null) {
        return MLXError.InvalidArray;
    }
    if (stream.ctx == null) {
        return MLXError.InvalidArray;
    }
    if (axis < -16 or axis > 16) {
        return MLXError.InvalidArray;
    }
    std.debug.assert(@intFromPtr(result) > 0);
    std.debug.assert(axis >= -16);
    std.debug.assert(axis <= 16);
    std.debug.assert(stream.ctx != null);
    std.debug.assert(logits.ctx != null);
    try op_call_checked(
        C.mlx_random_categorical(result, logits, axis, C.mlx_array_empty, stream),
        "mlx_random_categorical",
    );
}

pub fn tril(result: *Array, x: Array, offset: c_int, stream: Stream) MLXError!void {
    try op_call_checked(C.mlx_tril(result, x, offset, stream), "mlx_tril");
}

pub fn linspace(
    result: *Array,
    start: f64,
    stop: f64,
    num: c_int,
    dtype: C.mlx_dtype,
    stream: Stream,
) MLXError!void {
    try op_call_checked(C.mlx_linspace(result, start, stop, num, dtype, stream), "mlx_linspace");
}

pub fn arange(
    result: *Array,
    start: f64,
    stop: f64,
    step: f64,
    dtype: C.mlx_dtype,
    stream: Stream,
) MLXError!void {
    try op_call_checked(C.mlx_arange(result, start, stop, step, dtype, stream), "mlx_arange");
}

pub fn repeat(result: *Array, x: Array, repeats: c_int, axis: c_int, stream: Stream) MLXError!void {
    try op_call_checked(C.mlx_repeat(result, x, repeats, axis, stream), "mlx_repeat");
}

pub fn array_set(arr: *Array, src: Array) MLXError!void {
    try op_call_checked(C.mlx_array_set(arr, src), "mlx_array_set");
}

pub fn array_eval(arr: Array) !void {
    try op_call_checked(C.mlx_array_eval(arr), "mlx_array_eval");
}

pub fn array_free(arr: Array) void {
    discard(C.mlx_array_free(arr));
}

/// ============================================================================
/// Vector Operations
/// ============================================================================
pub fn einsum(
    result: *Array,
    arrays: []const Array,
    pattern: [*:0]const u8,
    stream: Stream,
) MLXError!void {
    if (arrays.len == 0) return MLXError.InvalidArray;
    std.debug.assert(arrays.len > 0);
    const operands = C.mlx_vector_array_new_data(arrays.ptr, arrays.len);
    defer discard(C.mlx_vector_array_free(operands));
    try op_status(C.mlx_einsum(result, pattern, operands, stream));
}

pub fn concatenate(
    result: *Array,
    arrays: []const Array,
    axis: c_int,
    stream: Stream,
) MLXError!void {
    if (arrays.len == 0) return MLXError.InvalidArray;
    std.debug.assert(arrays.len > 0);
    const vector_arrays = C.mlx_vector_array_new_data(arrays.ptr, arrays.len);
    defer discard(C.mlx_vector_array_free(vector_arrays));
    try op_status(C.mlx_concatenate(result, vector_arrays, axis, stream));
}

pub const ArrayRef = struct {
    ptr: *Array,
};

pub fn split(
    outputs: []const ArrayRef,
    a: Array,
    indices: []const c_int,
    axis: c_int,
    stream: Stream,
) MLXError!void {
    if (a.ctx == null) return MLXError.InvalidArray;
    if (outputs.len == 0) return MLXError.InvalidArray;
    if (indices.len == 0) return MLXError.InvalidArray;
    std.debug.assert(a.ctx != null);
    std.debug.assert(outputs.len > 0);
    std.debug.assert(indices.len > 0);
    var results = C.mlx_vector_array_new();
    defer discard(C.mlx_vector_array_free(results));
    try op_status(C.mlx_split(&results, a, indices.ptr, indices.len, axis, stream));
    for (outputs, 0..) |out_ref, i| {
        try op_status(C.mlx_vector_array_get(out_ref.ptr, results, i));
    }
}

pub fn split_equal_parts(
    outputs: []const ArrayRef,
    a: Array,
    num_splits: c_int,
    axis: c_int,
    stream: Stream,
) MLXError!void {
    if (a.ctx == null) return MLXError.InvalidArray;
    if (outputs.len == 0) return MLXError.InvalidArray;
    if (num_splits <= 0) return MLXError.InvalidArray;
    std.debug.assert(a.ctx != null);
    std.debug.assert(outputs.len > 0);
    std.debug.assert(num_splits > 0);
    var results = C.mlx_vector_array_new();
    defer discard(C.mlx_vector_array_free(results));
    try op_status(C.mlx_split_equal_parts(&results, a, num_splits, axis, stream));
    for (outputs, 0..) |out_ref, i| {
        try op_status(C.mlx_vector_array_get(out_ref.ptr, results, i));
    }
}

/// ============================================================================
/// Fast Operations
/// ============================================================================
pub fn fast_rope(
    result: *Array,
    x: Array,
    dims: c_int,
    traditional: bool,
    base: C.mlx_optional_float,
    scale: f32,
    offset: c_int,
    freqs: Array,
    s: Stream,
) MLXError!void {
    try op_status(C.mlx_fast_rope(result, x, dims, traditional, base, scale, offset, freqs, s));
}

pub fn fast_rms_norm(
    result: *Array,
    x: anytype,
    weight: anytype,
    eps: f32,
    stream: Stream,
) MLXError!void {
    const x_conv = array_converter(x);
    const weight_conv = array_converter(weight);
    defer {
        x_conv.deinit();
        weight_conv.deinit();
    }
    try op_status(C.mlx_fast_rms_norm(result, x_conv.arr, weight_conv.arr, eps, stream));
}

pub fn fast_layer_norm(
    result: *Array,
    x: anytype,
    weight: anytype,
    bias: anytype,
    eps: f32,
    stream: Stream,
) MLXError!void {
    const x_conv = array_converter(x);
    const weight_conv = array_converter(weight);
    const bias_conv = array_converter(bias);
    defer {
        x_conv.deinit();
        weight_conv.deinit();
        bias_conv.deinit();
    }
    try op_status(
        C.mlx_fast_layer_norm(
            result,
            x_conv.arr,
            weight_conv.arr,
            bias_conv.arr,
            eps,
            stream,
        ),
    );
}

/// ============================================================================
/// Stream Operations
/// ============================================================================
pub fn stream_free(stream: Stream) void {
    discard(C.mlx_stream_free(stream));
}

pub const default_cpu_stream_new = C.mlx_default_cpu_stream_new;
pub const default_gpu_stream_new = C.mlx_default_gpu_stream_new;

/// ============================================================================
/// File Operations
/// ============================================================================
pub fn load_safetensors(
    weights_hash: *std.StringHashMap(*Array),
    path_safetensors: [:0]const u8,
    stream: Stream,
) MLXError!void {
    if (path_safetensors.len == 0) return MLXError.FileNotFound;
    std.debug.assert(path_safetensors.len > 0);
    std.debug.assert(weights_hash.count() >= 0);
    const file = C.fopen(path_safetensors.ptr, "rb") orelse {
        return MLXError.FileNotFound;
    };
    defer discard(C.fclose(file));
    var weights = C.mlx_map_string_to_array_new();
    defer discard(C.mlx_map_string_to_array_free(weights));
    var meta = C.mlx_map_string_to_string_new();
    defer discard(C.mlx_map_string_to_string_free(meta));
    if (C.mlx_load_safetensors_file(&weights, &meta, file, stream) == 0) {} else {
        return MLXError.LoadWeightsFailed;
    }

    const iter = C.mlx_map_string_to_array_iterator_new(weights);
    defer discard(C.mlx_map_string_to_array_iterator_free(iter));
    var key: [*c]const u8 = undefined;
    var value = C.mlx_array_new();
    defer array_free(value);
    for (0..MAX_SAFE_TENSORS_ENTRIES) |_| {
        if (C.mlx_map_string_to_array_iterator_next(&key, &value, iter) == 0) {} else {
            break;
        }
        const key_str = std.mem.span(key);
        if (weights_hash.get(key_str)) |weight_ptr| {
            try array_set(weight_ptr, value);
            try op_status(C.mlx_array_eval(weight_ptr.*));
        } else {
            std.debug.print("\nKey not found in weights_hash: {s}\n", .{key_str});
            // return MLXError.KeyNotFoundInWeightsHash;
            // sinusoids in whisper.zig can be either loaded or created.
        }
    }
}

pub fn load_model_safetensors(
    weights_hash: *std.StringHashMap(*Array),
    path_dir: []const u8,
    stream: Stream,
) !void {
    if (path_dir.len == 0) return MLXError.FileNotFound;
    std.debug.assert(path_dir.len > 0);
    std.debug.assert(weights_hash.count() >= 0);
    const io = default_io();
    var buf: [1024]u8 = undefined;
    var dir = try std.Io.Dir.cwd().openDir(io, path_dir, .{ .iterate = true });
    defer dir.close(io);
    var iterator = dir.iterate();
    while (try iterator.next(io)) |entry| {
        if (entry.kind == .file) {
            const is_model_prefix = std.mem.startsWith(u8, entry.name, "model");
            const is_safetensors = std.mem.endsWith(u8, entry.name, ".safetensors");
            if (is_model_prefix and is_safetensors) {
                const path_file = try std.fmt.bufPrintZ(&buf, "{s}/{s}", .{ path_dir, entry.name });
                try load_safetensors(weights_hash, path_file, stream);
            }
        }
    }
}
pub const Safetensors = struct {
    const Self = @This();
    const MAX_PATH_LEN = 1024;
    file: ?*C.FILE,
    weights: MapStrArr,
    stream: Stream,

    pub fn load(path_safetensors: [:0]const u8, stream: Stream) MLXError!Self {
        const file = C.fopen(path_safetensors.ptr, "rb") orelse {
            return MLXError.FileNotFound;
        };
        var weights = C.mlx_map_string_to_array_new();
        errdefer {
            discard(C.mlx_map_string_to_array_free(weights));
            discard(C.fclose(file));
        }
        var meta = C.mlx_map_string_to_string_new();
        defer discard(C.mlx_map_string_to_string_free(meta));
        if (C.mlx_load_safetensors_file(&weights, &meta, file, stream) == 0) {} else {
            return MLXError.LoadWeightsFailed;
        }
        return Self{
            .file = file,
            .weights = weights,
            .stream = stream,
        };
    }

    pub fn unload(self: *Self, weights_hash: *std.StringHashMap(*Array)) MLXError!void {
        const mapsa_iter = C.mlx_map_string_to_array_iterator_new(self.weights);
        defer discard(C.mlx_map_string_to_array_iterator_free(mapsa_iter));
        var key: [*c]const u8 = undefined;
        var value = C.mlx_array_new();
        defer array_free(value);
        while (C.mlx_map_string_to_array_iterator_next(&key, &value, mapsa_iter) == 0) {
            const key_str = std.mem.span(key);
            if (weights_hash.get(key_str)) |weight_ptr| {
                try array_set(weight_ptr, value);
                try op_status(C.mlx_array_eval(weight_ptr.*));
            } else {
                std.debug.print("\nKey not found in weights_hash: {s}\n", .{key_str});
            }
        }
    }

    pub fn deinit(self: *Self) void {
        if (self.file) |file| {
            discard(C.fclose(file));
            self.file = null;
        }
        discard(C.mlx_map_string_to_array_free(self.weights));
    }
};

pub fn load_array(
    weight: *Array,
    name: []const u8,
    ext: ?[]const u8,
    weights_map: *const MapStrArr,
) MLXError!void {
    if (name.len == 0) return MLXError.InvalidArray;
    std.debug.assert(name.len > 0);
    var buf: [1024]u8 = undefined;
    var key: []const u8 = name;
    if (ext) |e| {
        if (e.len == 0) return MLXError.InvalidArray;
        key = try std.fmt.bufPrintZ(&buf, "{s}.{s}", .{ name, e });
    }
    if (key.len == 0) return MLXError.InvalidArray;
    try op_status(C.mlx_map_string_to_array_get(weight, weights_map.*, key.ptr));
    try op_status(C.mlx_array_eval(weight.*));
}

pub fn gelu(result: *Array, x: Array, stream: Stream) MLXError!void {
    std.debug.assert(x.ctx != null);
    var tmp = array_new();
    defer array_free(tmp);
    try divide(&tmp, x, float(@sqrt(2.0)), stream);
    try op_status(C.mlx_erf(&tmp, tmp, stream));
    try add(&tmp, tmp, float(1.0), stream);
    try multiply(&tmp, tmp, float(0.5), stream);
    try multiply(result, x, tmp, stream);
}

pub fn silu(result: *Array, x: Array, stream: Stream) MLXError!void {
    var tmp = array_new();
    defer array_free(tmp);
    try sigmoid(&tmp, x, stream);
    try multiply(result, tmp, x, stream);
}

/// ============================================================================
/// Utility Functions
/// ============================================================================
pub fn create_causal_mask(
    result: *Array,
    seq_len: c_int,
    offset: c_int,
    dtype: C.mlx_dtype,
    stream: Stream,
) MLXError!void {
    try ones(result, &[_]c_int{ seq_len, seq_len + offset }, C.MLX_BOOL, stream);
    try tril(result, result.*, offset, stream);
    try where(result, result.*, float(0.0), float(-std.math.inf(f32)), stream);
    try astype(result, result.*, dtype, stream);
    try array_eval(result.*);
}

pub fn print_array(msg: []const u8, arr: Array) void {
    if (msg.len == 0) {
        return;
    }
    if (arr.ctx == null) {
        return;
    }
    std.debug.assert(msg.len > 0);
    std.debug.assert(arr.ctx != null);
    var str = C.mlx_string_new();
    defer discard(C.mlx_string_free(str));
    discard(C.mlx_array_tostring(&str, arr));
    std.debug.print("{s}\n{s}\n", .{ msg, C.mlx_string_data(str) });
    std.debug.print("Shape: [", .{});
    const ndim = C.mlx_array_ndim(arr);
    const shape = C.mlx_array_shape(arr);
    for (0..ndim) |idx| {
        if (idx > 0) {
            std.debug.print(",", .{});
        }
        std.debug.print("{d}", .{shape[idx]});
    }
    std.debug.print("]\n", .{});
}

pub fn print_map_str(msg: []const u8, map: *C.mlx_map_string_to_string) MLXError!void {
    if (msg.len == 0) return MLXError.InvalidArray;
    std.debug.assert(msg.len > 0);
    const map_iter = C.mlx_map_string_to_string_iterator_new(map.*);
    defer discard(C.mlx_map_string_to_string_iterator_free(map_iter));
    var key: [*c]const u8 = undefined;
    var value: [*c]const u8 = undefined;
    std.debug.print("\n{s}:\n", .{msg});
    for (0..MAX_SAFE_TENSORS_ENTRIES) |_| {
        if (C.mlx_map_string_to_string_iterator_next(&key, &value, map_iter) == 0) {} else {
            break;
        }
        std.debug.print("  {s}: {s}\n", .{ key, value });
    }
}

pub fn print_map_arr(msg: []const u8, map: *const MapStrArr) MLXError!void {
    if (msg.len == 0) return MLXError.InvalidArray;
    std.debug.assert(msg.len > 0);
    const map_iter = C.mlx_map_string_to_array_iterator_new(map.*);
    defer discard(C.mlx_map_string_to_array_iterator_free(map_iter));
    var key: [*c]const u8 = undefined;
    var value = C.mlx_array_new();
    defer array_free(value);
    std.debug.print("\n{s}:\n", .{msg});
    for (0..MAX_SAFE_TENSORS_ENTRIES) |_| {
        if (C.mlx_map_string_to_array_iterator_next(&key, &value, map_iter) == 0) {} else {
            break;
        }
        std.debug.assert(key != null);
        std.debug.assert(value.ctx != null);
        const ndim = C.mlx_array_ndim(value);
        const shape = C.mlx_array_shape(value);
        std.debug.print("  {s}: shape=[", .{key});
        for (0..ndim) |idx| {
            if (idx > 0) {
                std.debug.print(", ", .{});
            }
            std.debug.print("{d}", .{shape[idx]});
        }
        std.debug.print("]\n", .{});
    }
}

/// ============================================================================
/// Helper Functions
/// ============================================================================
pub const MLXError = error{
    OperationFailed,
    InvalidArray,
    DeviceError,
    FileNotFound,
    LoadWeightsFailed,
    KeyNotFoundInWeightsHash,
    OutOfMemory,
    NoSpaceLeft,
};

pub fn op_status(result: c_int) MLXError!void {
    if (result == 0) return;
    return MLXError.OperationFailed;
}

pub fn op_call_checked(result: c_int, comptime func_name: []const u8) MLXError!void {
    if (func_name.len == 0) return MLXError.OperationFailed;
    std.debug.assert(func_name.len > 0);
    if (result == 0) {
        std.debug.assert(result == 0);
        return;
    }
    std.log.err("MLX operation '{s}' failed with code {d}", .{ func_name, result });
    return MLXError.OperationFailed;
}

fn define_unary_op(comptime c_func_name: []const u8) fn (*Array, anytype, Stream) MLXError!void {
    if (c_func_name.len == 0) {
        return struct {
            fn impl(_: *Array, _: anytype, _: Stream) MLXError!void {
                return MLXError.InvalidArray;
            }
        }.impl;
    }
    std.debug.assert(c_func_name.len > 0);
    return struct {
        fn impl(result: *Array, a: anytype, stream: Stream) MLXError!void {
            const a_conv = array_converter(a);
            defer a_conv.deinit();
            if (a_conv.arr.ctx == null) return MLXError.InvalidArray;
            std.debug.assert(a_conv.arr.ctx != null);
            try op_call_checked(@field(C, c_func_name)(result, a_conv.arr, stream), c_func_name);
        }
    }.impl;
}

fn define_binary_op(
    comptime c_func_name: []const u8,
) fn (*Array, anytype, anytype, Stream) MLXError!void {
    if (c_func_name.len == 0) {
        return struct {
            fn impl(_: *Array, _: anytype, _: anytype, _: Stream) MLXError!void {
                return MLXError.InvalidArray;
            }
        }.impl;
    }
    std.debug.assert(c_func_name.len > 0);
    const BinaryOpImpl = struct {
        fn impl(result: *Array, a: anytype, b: anytype, stream: Stream) MLXError!void {
            const a_conv = array_converter(a);
            const b_conv = array_converter(b);
            defer {
                a_conv.deinit();
                b_conv.deinit();
            }
            if (a_conv.arr.ctx == null or b_conv.arr.ctx == null) return MLXError.InvalidArray;
            std.debug.assert(a_conv.arr.ctx != null);
            std.debug.assert(b_conv.arr.ctx != null);
            try op_call_checked(
                @field(C, c_func_name)(result, a_conv.arr, b_conv.arr, stream),
                c_func_name,
            );
        }
    };
    return BinaryOpImpl.impl;
}

fn define_reduce_all_op(
    comptime c_func_name: []const u8,
) fn (*Array, Array, bool, Stream) MLXError!void {
    if (c_func_name.len == 0) {
        return struct {
            fn impl(_: *Array, _: Array, _: bool, _: Stream) MLXError!void {
                return MLXError.InvalidArray;
            }
        }.impl;
    }
    std.debug.assert(c_func_name.len > 0);
    return struct {
        fn impl(result: *Array, x: Array, keepdims: bool, stream: Stream) MLXError!void {
            if (x.ctx == null) return MLXError.InvalidArray;
            std.debug.assert(x.ctx != null);
            try op_call_checked(@field(C, c_func_name)(result, x, keepdims, stream), c_func_name);
        }
    }.impl;
}

fn define_reduce_axes_op(
    comptime c_func_name: []const u8,
) fn (*Array, Array, []const c_int, bool, Stream) MLXError!void {
    if (c_func_name.len == 0) {
        return struct {
            fn impl(_: *Array, _: Array, _: []const c_int, _: bool, _: Stream) MLXError!void {
                return MLXError.InvalidArray;
            }
        }.impl;
    }
    std.debug.assert(c_func_name.len > 0);
    return struct {
        fn impl(
            result: *Array,
            x: Array,
            axes: []const c_int,
            keepdims: bool,
            stream: Stream,
        ) MLXError!void {
            if (x.ctx == null) return MLXError.InvalidArray;
            if (axes.len == 0) return MLXError.InvalidArray;
            std.debug.assert(x.ctx != null);
            std.debug.assert(axes.len > 0);
            try op_call_checked(
                @field(C, c_func_name)(result, x, axes.ptr, axes.len, keepdims, stream),
                c_func_name,
            );
        }
    }.impl;
}

fn define_fft_op(
    comptime c_func_name: []const u8,
) fn (*Array, Array, c_int, c_int, Stream) MLXError!void {
    if (c_func_name.len == 0) {
        return struct {
            fn impl(_: *Array, _: Array, _: c_int, _: c_int, _: Stream) MLXError!void {
                return MLXError.InvalidArray;
            }
        }.impl;
    }
    std.debug.assert(c_func_name.len > 0);
    return struct {
        fn impl(result: *Array, x: Array, n: c_int, axis: c_int, stream: Stream) MLXError!void {
            if (x.ctx == null) return MLXError.InvalidArray;
            if (n <= 0) return MLXError.InvalidArray;
            if (axis >= -16 and axis <= 16) {
                std.debug.assert(axis >= -16);
                std.debug.assert(axis <= 16);
            } else {
                return MLXError.InvalidArray;
            }
            std.debug.assert(x.ctx != null);
            std.debug.assert(n > 0);
            try op_call_checked(@field(C, c_func_name)(result, x, n, axis, stream), c_func_name);
        }
    }.impl;
}

fn define_fftn_op(
    comptime c_func_name: []const u8,
) fn (*Array, Array, []const c_int, []const c_int, Stream) MLXError!void {
    if (c_func_name.len == 0) {
        return struct {
            fn impl(
                _: *Array,
                _: Array,
                _: []const c_int,
                _: []const c_int,
                _: Stream,
            ) MLXError!void {
                return MLXError.InvalidArray;
            }
        }.impl;
    }
    std.debug.assert(c_func_name.len > 0);
    return struct {
        fn impl(
            result: *Array,
            x: Array,
            n: []const c_int,
            axes: []const c_int,
            stream: Stream,
        ) MLXError!void {
            if (x.ctx == null) return MLXError.InvalidArray;
            if (n.len == 0) return MLXError.InvalidArray;
            if (n.len == axes.len) {
                std.debug.assert(n.len == axes.len);
            } else {
                return MLXError.InvalidArray;
            }
            std.debug.assert(x.ctx != null);
            std.debug.assert(n.len > 0);
            std.debug.assert(n.len == axes.len);
            try op_call_checked(
                @field(C, c_func_name)(result, x, n.ptr, n.len, axes.ptr, axes.len, stream),
                c_func_name,
            );
        }
    }.impl;
}

fn array_converter(value: anytype) ArrayHandle {
    const T = @TypeOf(value);
    if (T == Array) {
        return ArrayHandle.init(value, false);
    } else if (T == FloatArg) {
        return ArrayHandle.init(C.mlx_array_new_float(value.value), true);
    } else if (T == IntArg) {
        return ArrayHandle.init(C.mlx_array_new_int(value.value), true);
    } else if (T == BoolArg) {
        return ArrayHandle.init(C.mlx_array_new_bool(value.value), true);
    } else {
        @compileError("Unsupported type: " ++ @typeName(T));
    }
}

const ArrayHandle = struct {
    arr: Array,
    owned: bool,

    pub fn init(array: Array, owned: bool) ArrayHandle {
        return .{ .arr = array, .owned = owned };
    }

    pub fn deinit(self: ArrayHandle) void {
        if (self.owned) {
            array_free(self.arr);
        }
    }
};

const FloatArg = struct {
    value: f32,

    pub fn init(value: f32) FloatArg {
        return .{ .value = value };
    }
};

const IntArg = struct {
    value: c_int,

    pub fn init(value: anytype) IntArg {
        return .{ .value = @intCast(value) };
    }
};

const BoolArg = struct {
    value: bool,

    pub fn init(value: bool) BoolArg {
        return .{ .value = value };
    }
};

pub const float = FloatArg.init;
pub const int = IntArg.init;
pub const bool_value = BoolArg.init;

/// ============================================================================
/// Experimental Array Shape Operations
/// ============================================================================
pub const RepeatPatternDims = struct {
    repeat: c_int,
};

const repeat_bhld_pattern = "b h l d -> b (repeat h) l d";

pub fn repeat_pattern(
    result: *Array,
    x: Array,
    pattern: []const u8,
    dim_values: RepeatPatternDims,
    stream: Stream,
) MLXError!void {
    if (@intFromPtr(result) == 0) return MLXError.InvalidArray;
    if (x.ctx == null) return MLXError.InvalidArray;
    if (pattern.len == 0) return MLXError.InvalidArray;
    if (dim_values.repeat <= 0) return MLXError.InvalidArray;

    const supported_pattern = std.mem.eql(u8, pattern, repeat_bhld_pattern);
    if (supported_pattern == false) return MLXError.InvalidArray;

    std.debug.assert(@intFromPtr(result) > 0);
    std.debug.assert(x.ctx != null);
    std.debug.assert(pattern.len > 0);
    std.debug.assert(dim_values.repeat > 0);
    std.debug.assert(supported_pattern);

    try op_call_checked(
        C.mlx_repeat(result, x, dim_values.repeat, 1, stream),
        "repeat_pattern",
    );
}

pub fn reshape_to_heads(
    result: *Array,
    x: Array,
    h: c_int,
    d: c_int,
    stream: Stream,
) MLXError!void {
    try array_set(result, x);
    const shape = [_]c_int{ h, d };
    try op_call_checked(
        C.mlx_unflatten(result, result.*, 2, &shape, shape.len, stream),
        "reshape_to_heads.unflatten",
    );
    try op_call_checked(
        C.mlx_swapaxes(result, result.*, 1, 2, stream),
        "reshape_to_heads.swap",
    );
}

pub fn reshape_from_heads(
    result: *Array,
    x: Array,
    stream: Stream,
) MLXError!void {
    try array_set(result, x);
    try op_call_checked(
        C.mlx_swapaxes(result, result.*, 1, 2, stream),
        "reshape_from_heads.swap",
    );
    try op_call_checked(
        C.mlx_flatten(result, result.*, 2, 3, stream),
        "reshape_from_heads.flatten",
    );
}
