const std = @import("std");
const build_options = @import("build_options");
const libmlx = @import("libmlx");

const download_model = libmlx.utils.download_model;
const Tokenizer = libmlx.tokenizer.Tokenizer;
const LlamaTransformer = libmlx.llama.Transformer;
const PhiTransformer = libmlx.phi.Transformer;
const QwenTransformer = libmlx.qwen.Transformer;
const SamplingConfig = libmlx.mlx.SamplingConfig;
const default_model_assets_dir = "src/assets/models";

const ChatConfig = struct {
    model_type: []const u8,
    model_name: []const u8,
    chat_format: ?[]const u8,
    chat_inputs: []const []const u8,
    num_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: usize,
};

const TransformerUnion = union(enum) {
    llama: LlamaTransformer,
    phi: PhiTransformer,
    qwen: QwenTransformer,

    fn init(allocator: std.mem.Allocator, model_type: []const u8, model_name: []const u8) !TransformerUnion {
        if (std.mem.eql(u8, model_type, "llama")) return .{ .llama = try LlamaTransformer.init(allocator, model_name) };
        if (std.mem.eql(u8, model_type, "phi")) return .{ .phi = try PhiTransformer.init(allocator, model_name) };
        if (std.mem.eql(u8, model_type, "qwen")) return .{ .qwen = try QwenTransformer.init(allocator, model_name) };
        return error.UnsupportedModelType;
    }

    fn deinit(self: *TransformerUnion) void {
        switch (self.*) {
            inline else => |*t| t.deinit(),
        }
    }

    fn generate(
        self: *TransformerUnion,
        input: []const u32,
        num_tokens: usize,
        sampling: SamplingConfig,
    ) ![]u32 {
        return switch (self.*) {
            inline else => |*t| t.generateWithSampling(input, num_tokens, sampling),
        };
    }
};

const ModelPathResolution = struct {
    path: []u8,
    should_download: bool,
};

fn modelAssetsDir() []const u8 {
    return build_options.model_assets_dir orelse default_model_assets_dir;
}

fn pathExists(path: []const u8) bool {
    if (path.len == 0) {
        return false;
    }
    std.debug.assert(path.len > 0);
    const io = std.Io.Threaded.global_single_threaded.io();
    std.Io.Dir.cwd().access(io, path, .{}) catch {
        return false;
    };
    return true;
}

fn hasModelFiles(allocator: std.mem.Allocator, model_path: []const u8) !bool {
    if (!pathExists(model_path)) {
        return false;
    }

    const config_path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{model_path});
    defer allocator.free(config_path);
    if (!pathExists(config_path)) {
        return false;
    }

    const tokenizer_path = try std.fmt.allocPrint(allocator, "{s}/tokenizer.json", .{model_path});
    defer allocator.free(tokenizer_path);
    if (!pathExists(tokenizer_path)) {
        return false;
    }

    const weights_path = try std.fmt.allocPrint(allocator, "{s}/model.safetensors", .{model_path});
    defer allocator.free(weights_path);
    return pathExists(weights_path);
}

fn resolveModelPath(allocator: std.mem.Allocator, model_name: []const u8) !ModelPathResolution {
    if (try hasModelFiles(allocator, model_name)) {
        return .{
            .path = try allocator.dupe(u8, model_name),
            .should_download = false,
        };
    }

    const model_assets_path = try std.fmt.allocPrint(
        allocator,
        "{s}/{s}",
        .{ modelAssetsDir(), model_name },
    );
    if (try hasModelFiles(allocator, model_assets_path)) {
        return .{
            .path = model_assets_path,
            .should_download = false,
        };
    }
    allocator.free(model_assets_path);
    return .{
        .path = try allocator.dupe(u8, model_name),
        .should_download = true,
    };
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var args = init.minimal.args.iterate();
    _ = args.next();

    var chat_inputs = std.array_list.Managed([]const u8).init(allocator);
    defer chat_inputs.deinit();

    var preset_name: ?[]const u8 = null;
    var model_type_override: ?[]const u8 = null;
    var model_name_override: ?[]const u8 = null;
    var num_tokens_override: ?usize = null;
    var temperature_override: ?f32 = null;
    var top_p_override: ?f32 = null;
    var top_k_override: ?usize = null;
    var chat_format_override: ?[]const u8 = null;

    while (args.next()) |arg| {
        if (!std.mem.startsWith(u8, arg, "--")) {
            try chat_inputs.append(arg);
            continue;
        }

        if (std.mem.eql(u8, arg, "--help")) {
            printUsage();
            return;
        }

        if (std.mem.indexOfScalar(u8, arg, '=')) |idx| {
            const key = arg[2..idx];
            const value = arg[idx + 1 ..];
            if (std.mem.eql(u8, key, "config")) preset_name = value;
            if (std.mem.eql(u8, key, "model-type")) model_type_override = value;
            if (std.mem.eql(u8, key, "model-name")) model_name_override = value;
            if (std.mem.eql(u8, key, "format")) chat_format_override = value;
            if (std.mem.eql(u8, key, "max")) num_tokens_override = try std.fmt.parseInt(usize, value, 10);
            if (std.mem.eql(u8, key, "temperature")) temperature_override = try std.fmt.parseFloat(f32, value);
            if (std.mem.eql(u8, key, "top-p")) top_p_override = try std.fmt.parseFloat(f32, value);
            if (std.mem.eql(u8, key, "top-k")) top_k_override = try std.fmt.parseInt(usize, value, 10);
        }
    }

    var config = presetConfig(preset_name orelse build_options.config orelse "qwq");
    if (build_options.model_type) |v| config.model_type = v;
    if (build_options.model_name) |v| config.model_name = v;
    if (build_options.format) |v| config.chat_format = v;
    if (build_options.max) |v| config.num_tokens = v;
    if (build_options.temperature) |v| config.temperature = v;
    if (build_options.top_p) |v| config.top_p = v;
    if (build_options.top_k) |v| config.top_k = v;

    if (model_type_override) |v| config.model_type = v;
    if (model_name_override) |v| config.model_name = v;
    if (chat_format_override) |v| config.chat_format = v;
    if (num_tokens_override) |v| config.num_tokens = v;
    if (temperature_override) |v| config.temperature = v;
    if (top_p_override) |v| config.top_p = v;
    if (top_k_override) |v| config.top_k = v;
    if (chat_inputs.items.len > 0) config.chat_inputs = chat_inputs.items;

    std.debug.print("\n=== {s} ({s}) ===\n", .{ config.model_type, config.model_name });

    const model_path = try resolveModelPath(allocator, config.model_name);
    defer allocator.free(model_path.path);
    if (model_path.should_download) {
        try download_model(allocator, "mlx-community", config.model_name);
    }

    var tokenizer = try Tokenizer.init(allocator, model_path.path);
    defer tokenizer.deinit();

    const input_ids = try tokenizer.encode_chat(config.chat_format, config.chat_inputs);
    defer allocator.free(input_ids);

    var transformer = try TransformerUnion.init(allocator, config.model_type, model_path.path);
    defer transformer.deinit();

    const output_ids = try transformer.generate(input_ids, config.num_tokens, .{
        .temperature = config.temperature,
        .top_p = config.top_p,
        .top_k = config.top_k,
    });
    defer allocator.free(output_ids);

    const input_str = try tokenizer.decode(input_ids);
    defer allocator.free(input_str);

    const output_str = try tokenizer.decode(output_ids);
    defer allocator.free(output_str);

    std.debug.print("\nInput:\n{s}\n\nOutput:\n{s}\n", .{ input_str, output_str });
}

fn presetConfig(name: []const u8) ChatConfig {
    if (std.mem.eql(u8, name, "phi")) {
        return .{
            .model_type = "phi",
            .model_name = "phi-4-2bit",
            .chat_format =
            \\<|im_start|>system<|im_sep|>
            \\You are a precise coding assistant.<|im_end|>
            \\<|im_start|>user<|im_sep|>
            \\{s}<|im_end|>
            \\<|im_start|>assistant<|im_sep|>
            \\
            ,
            .chat_inputs = &.{"Explain Zig comptime in 3 bullet points."},
            .num_tokens = 30,
            .temperature = 0.0,
            .top_p = 1.0,
            .top_k = 0,
        };
    }

    if (std.mem.eql(u8, name, "llama")) {
        return .{
            .model_type = "llama",
            .model_name = "Llama-3.2-1B-Instruct-4bit",
            .chat_format =
            \\<|begin_of_text|><|start_header_id|>user<|end_header_id|>
            \\
            \\{s}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            \\
            \\
            ,
            .chat_inputs = &.{"Write a Zig function that checks if a number is prime."},
            .num_tokens = 30,
            .temperature = 0.0,
            .top_p = 1.0,
            .top_k = 0,
        };
    }

    return .{
        .model_type = "qwen",
        .model_name = "QwQ-32B-3bit",
        .chat_format =
        \\<|im_start|>user
        \\{s}<|im_end|>
        \\<|im_start|>assistant
        \\<think>
        \\
        ,
        .chat_inputs = &.{"Write a Zig function that checks if a number is prime."},
        .num_tokens = if (build_options.max) |v| v else 30,
        .temperature = 0.0,
        .top_p = 1.0,
        .top_k = 0,
    };
}

fn printUsage() void {
    const usage =
        \\Options:
        \\  --config=CONFIG         Config preset: qwq, llama, phi
        \\  --format=FORMAT         Custom chat format template string
        \\  --model-type=TYPE       Model type override
        \\  --model-name=NAME       Model name override
        \\  --max=N                 Maximum number of tokens to generate
        \\  --temperature=T         Sampling temperature (0 = greedy)
        \\  --top-p=P               Nucleus sampling threshold (0, 1]
        \\  --top-k=K               Keep K highest-logit tokens (0 = off)
        \\  --help                  Show this help
        \\
    ;
    std.debug.print("{s}", .{usage});
}
