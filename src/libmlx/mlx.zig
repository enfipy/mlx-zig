pub const core = @import("mlx_core.zig");
pub const nn = @import("mlx_nn.zig");
pub const runtime = @import("mlx_runtime.zig");

pub const C = core.C;

pub const Array = core.Array;
pub const Stream = core.Stream;
pub const OptionalFloat = core.OptionalFloat;
pub const Safetensors = core.Safetensors;

pub const FLOAT16 = core.FLOAT16;
pub const FLOAT32 = core.FLOAT32;
pub const BFLOAT16 = core.BFLOAT16;
pub const UINT32 = core.UINT32;

pub const add = core.add;
pub const subtract = core.subtract;
pub const multiply = core.multiply;
pub const divide = core.divide;
pub const power = core.power;
pub const greater = core.greater;
pub const less_equal = core.less_equal;
pub const logical_or = core.logical_or;
pub const logical_not = core.logical_not;
pub const sigmoid = core.sigmoid;
pub const sin = core.sin;
pub const cos = core.cos;
pub const exp = core.exp;
pub const arange = core.arange;
pub const abs = core.abs;
pub const square = core.square;
pub const log10 = core.log10;
pub const matmul = core.matmul;
pub const maximum = core.maximum;
pub const where = core.where;
pub const take = core.take;
pub const pad = core.pad;
pub const slice = core.slice;
pub const as_strided = core.as_strided;
pub const expand_dims = core.expand_dims;
pub const softmax = core.softmax;
pub const argmax = core.argmax;
pub const argsort = core.argsort;
pub const cumsum = core.cumsum;
pub const all_max = core.all_max;
pub const topk = core.topk;
pub const rfft = core.rfft;
pub const concatenate = core.concatenate;
pub const split = core.split;
pub const split_equal_parts = core.split_equal_parts;
pub const einsum = core.einsum;
pub const take_along_axis = core.take_along_axis;
pub const astype = core.astype;
pub const random_categorical = core.random_categorical;
pub const fast_rope = core.fast_rope;
pub const gelu = core.gelu;
pub const silu = core.silu;

pub const array_new = core.array_new;
pub const array_new_float = core.array_new_float;
pub const array_new_data = core.array_new_data;
pub const array_set_data = core.array_set_data;
pub const array_dim = core.array_dim;
pub const array_eval = core.array_eval;
pub const array_free = core.array_free;

pub const create_causal_mask = core.create_causal_mask;
pub const repeat_pattern = core.repeat_pattern;
pub const reshape_to_heads = core.reshape_to_heads;
pub const reshape_from_heads = core.reshape_from_heads;

pub const load_model_safetensors = core.load_model_safetensors;
pub const load_array = core.load_array;

pub const item = core.item;
pub const op_status = core.op_status;
pub const float = core.float;
pub const int = core.int;

pub const QuantConfig = nn.QuantConfig;
pub const MLXConfig = nn.MLXConfig;
pub const Module = nn.Module;
pub const Weight = nn.Weight;
pub const Linear = nn.Linear;
pub const Embedding = nn.Embedding;
pub const RMSNorm = nn.RMSNorm;
pub const LayerNorm = nn.LayerNorm;
pub const Conv1d = nn.Conv1d;
pub const RoPE = nn.RoPE;

pub const ModelConfig = runtime.ModelConfig;
pub const SamplingConfig = runtime.SamplingConfig;
pub const Model = runtime.Model;
pub const Transformer = runtime.Transformer;
pub const KVCache = runtime.KVCache;
pub const Cache = runtime.Cache;

pub const API = @This();
