const std = @import("std");

const LlmOptions = struct {
    config: ?[]const u8,
    format: ?[]const u8,
    model_type: ?[]const u8,
    model_name: ?[]const u8,
    max: ?usize,
    temperature: ?f32,
    top_p: ?f32,
    top_k: ?usize,
    model_assets_dir: ?[]const u8,
    fixture_dir: ?[]const u8,

    fn fromOptions(b: *std.Build) LlmOptions {
        return .{
            .config = b.option([]const u8, "config", "Config: phi, llama, qwen, olympic"),
            .model_type = b.option([]const u8, "model-type", "Model-type: phi, llama, qwen"),
            .model_name = b.option([]const u8, "model-name", "Model-name"),
            .format = b.option([]const u8, "format", "Chat format"),
            .max = b.option(usize, "max", "Maximum number of tokens to generate"),
            .temperature = b.option(f32, "temperature", "Sampling temperature (0 = greedy)"),
            .top_p = b.option(f32, "top-p", "Nucleus sampling threshold (0, 1]"),
            .top_k = b.option(usize, "top-k", "Keep K highest-logit tokens (0 = off)"),
            .model_assets_dir = b.option([]const u8, "model-assets-dir", "Path to model asset files"),
            .fixture_dir = b.option([]const u8, "fixture-dir", "Path to test fixtures"),
        };
    }

    fn createBuildOptions(self: LlmOptions, b: *std.Build) *std.Build.Step.Options {
        const options_pkg = b.addOptions();
        options_pkg.addOption(?[]const u8, "config", self.config);
        options_pkg.addOption(?[]const u8, "format", self.format);
        options_pkg.addOption(?[]const u8, "model_type", self.model_type);
        options_pkg.addOption(?[]const u8, "model_name", self.model_name);
        options_pkg.addOption(?usize, "max", self.max);
        options_pkg.addOption(?f32, "temperature", self.temperature);
        options_pkg.addOption(?f32, "top_p", self.top_p);
        options_pkg.addOption(?usize, "top_k", self.top_k);
        options_pkg.addOption(?[]const u8, "model_assets_dir", self.model_assets_dir);
        options_pkg.addOption(?[]const u8, "fixture_dir", self.fixture_dir);
        return options_pkg;
    }
};

const Dependencies = struct {
    mlx_c_dep: *std.Build.Dependency,
    mlx_core_lib: *std.Build.Step.Compile,
    mlx_c_lib: *std.Build.Step.Compile,
    install_step: *std.Build.Step,
};

const MLXC_CPP_FLAGS = [_][]const u8{
    "-std=c++17",
    "-fPIC",
    "-D_GLIBCXX_USE_CXX11_ABI=1",
    "-U__ARM_FEATURE_BF16",
    "-fexceptions",
};

const mlxc_sources = [_][]const u8{
    "mlx/c/array.cpp",
    "mlx/c/closure.cpp",
    "mlx/c/compile.cpp",
    "mlx/c/device.cpp",
    "mlx/c/distributed.cpp",
    "mlx/c/distributed_group.cpp",
    "mlx/c/error.cpp",
    "mlx/c/fast.cpp",
    "mlx/c/fft.cpp",
    "mlx/c/io.cpp",
    "mlx/c/linalg.cpp",
    "mlx/c/map.cpp",
    "mlx/c/metal.cpp",
    "mlx/c/ops.cpp",
    "mlx/c/random.cpp",
    "mlx/c/stream.cpp",
    "mlx/c/string.cpp",
    "mlx/c/transforms.cpp",
    "mlx/c/transforms_impl.cpp",
    "mlx/c/vector.cpp",
};

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const mlx_build_metal = b.option(bool, "mlx-build-metal", "Build MLX-C with Metal backend") orelse false;

    const deps = try setupDependencies(b, target, optimize, mlx_build_metal);
    const llm_options = LlmOptions.fromOptions(b);
    const build_options_pkg = llm_options.createBuildOptions(b);
    const libmlx_module = b.createModule(.{
        .root_source_file = b.path("src/libmlx/libmlx.zig"),
        .target = target,
        .optimize = optimize,
    });
    configureLibModule(libmlx_module, deps);

    const llm_module = b.createModule(.{
        .root_source_file = b.path("src/tools/llm.zig"),
        .target = target,
        .optimize = optimize,
    });
    llm_module.addImport("libmlx", libmlx_module);
    llm_module.addOptions("build_options", build_options_pkg);

    const llm_exe = b.addExecutable(.{
        .name = "llm",
        .root_module = llm_module,
    });
    configureExecutable(llm_exe, deps);
    b.installArtifact(llm_exe);

    const whisper_module = b.createModule(.{
        .root_source_file = b.path("src/tools/whisper.zig"),
        .target = target,
        .optimize = optimize,
    });
    whisper_module.addOptions("build_options", build_options_pkg);

    const whisper_exe = b.addExecutable(.{
        .name = "whisper",
        .root_module = whisper_module,
    });
    whisper_exe.root_module.addImport("libmlx", libmlx_module);
    configureExecutable(whisper_exe, deps);
    b.installArtifact(whisper_exe);

    const runner_exe = b.addExecutable(.{
        .name = "mlx",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/mlx/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    configureExecutable(runner_exe, deps);
    b.installArtifact(runner_exe);

    addRunSteps(b, llm_exe, whisper_exe, runner_exe);
    const test_step = addTestSteps(b, deps, target, optimize);
    addTigercheckSteps(b, test_step);
    addBuildMlxCoreStep(b, deps);
}

fn addRunSteps(
    b: *std.Build,
    llm_exe: *std.Build.Step.Compile,
    whisper_exe: *std.Build.Step.Compile,
    runner_exe: *std.Build.Step.Compile,
) void {
    const llm_run = b.addRunArtifact(llm_exe);
    if (b.args) |args| llm_run.addArgs(args);
    const run_llm = b.step("run-llm", "Run LLM app");
    run_llm.dependOn(&llm_run.step);

    const whisper_run = b.addRunArtifact(whisper_exe);
    if (b.args) |args| whisper_run.addArgs(args);
    const run_whisper = b.step("run-whisper", "Run TTS app");
    run_whisper.dependOn(&whisper_run.step);

    const app_run = b.addRunArtifact(runner_exe);
    if (b.args) |args| app_run.addArgs(args);
    const run = b.step("run", "Run default app");
    run.dependOn(&app_run.step);
}

fn addTestSteps(
    b: *std.Build,
    deps: Dependencies,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Step {
    const lib_tests_module = b.createModule(.{
        .root_source_file = b.path("src/libmlx/libmlx.zig"),
        .target = target,
        .optimize = optimize,
    });

    const lib_tests = b.addTest(.{
        .root_module = lib_tests_module,
    });
    configureExecutable(lib_tests, deps);

    const app_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/mlx/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    configureExecutable(app_tests, deps);

    const run_lib_tests = b.addRunArtifact(lib_tests);
    run_lib_tests.stdio = .zig_test;

    const test_step = b.step("test", "Run all tests");
    test_step.dependOn(&run_lib_tests.step);
    test_step.dependOn(&app_tests.step);
    return test_step;
}

fn addTigercheckSteps(b: *std.Build, test_step: *std.Build.Step) void {
    const style_path = b.option([]const u8, "style-path", "Path for tigercheck analysis") orelse "src/libmlx";
    const tigercheck_target = b.option([]const u8, "tigercheck-target", "Path for tigercheck target") orelse style_path;
    const tigercheck_exe = b.option([]const u8, "tigercheck-exe", "Path to tigercheck binary") orelse resolveTigercheckPath(b);

    const check_cmd = b.addSystemCommand(&.{ tigercheck_exe, "--format", "text", tigercheck_target });
    const check_step = b.step("check", "Compile + tigercheck");
    check_step.dependOn(test_step);
    check_step.dependOn(&check_cmd.step);

    const check_strict_cmd = b.addSystemCommand(&.{ tigercheck_exe, tigercheck_target });
    const check_strict_step = b.step("check-strict", "Run strict tigercheck lane");
    check_strict_step.dependOn(test_step);
    check_strict_step.dependOn(&check_strict_cmd.step);
}

fn resolveTigercheckPath(b: *std.Build) []const u8 {
    const host = b.graph.host.result;
    const dep_name = tigercheckDepName(host);
    if (dep_name.len == 0) return "tigercheck";
    const dep = b.lazyDependency(dep_name, .{}) orelse return "tigercheck";
    return switch (host.os.tag) {
        .windows => dep.path("tigercheck.exe").getPath(b),
        else => dep.path("tigercheck").getPath(b),
    };
}

fn tigercheckDepName(host: std.Target) []const u8 {
    return switch (host.os.tag) {
        .linux => switch (host.cpu.arch) {
            .x86_64 => "tigercheck_x86_64_linux",
            .aarch64 => "tigercheck_aarch64_linux",
            else => "",
        },
        .macos => switch (host.cpu.arch) {
            .aarch64 => "tigercheck_aarch64_macos",
            .x86_64 => "tigercheck_x86_64_macos",
            else => "",
        },
        .windows => switch (host.cpu.arch) {
            .x86_64 => "tigercheck_x86_64_windows",
            .aarch64 => "tigercheck_aarch64_windows",
            else => "",
        },
        else => "",
    };
}

fn setupDependencies(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    mlx_build_metal: bool,
) !Dependencies {
    const mlx_core_dep = b.dependency("zig_build_mlx", .{
        .target = target,
        .@"build-tests" = false,
        .@"build-metal" = mlx_build_metal,
        .@"build-cpu" = true,
        .@"build-gguf" = true,
        .@"build-safetensors" = true,
        .@"metal-jit" = false,
    });
    const mlx_core_lib = mlx_core_dep.artifact("mlx");

    const mlx_c_dep = b.dependency("mlx_c", .{
        .target = target,
        .optimize = optimize,
    });

    const mlx_c_lib = b.addLibrary(.{
        .linkage = .static,
        .name = "mlxc",
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
        }),
    });
    mlx_c_lib.root_module.addIncludePath(mlx_c_dep.path("."));
    mlx_c_lib.root_module.addIncludePath(mlx_core_dep.path("."));
    mlx_c_lib.root_module.addCSourceFiles(.{
        .root = mlx_c_dep.path("."),
        .files = &mlxc_sources,
        .flags = &MLXC_CPP_FLAGS,
    });
    mlx_c_lib.root_module.linkLibrary(mlx_core_lib);
    mlx_c_lib.root_module.linkSystemLibrary("c++", .{});

    const install_step = b.step("install-mlx-c", "Build MLX-C wrapper and MLX core");
    install_step.dependOn(&mlx_c_lib.step);

    return .{
        .mlx_c_dep = mlx_c_dep,
        .mlx_core_lib = mlx_core_lib,
        .mlx_c_lib = mlx_c_lib,
        .install_step = install_step,
    };
}

fn configureExecutable(
    exe: *std.Build.Step.Compile,
    deps: Dependencies,
) void {
    exe.step.dependOn(deps.install_step);
    exe.root_module.addSystemIncludePath(deps.mlx_c_dep.path("."));
    exe.root_module.linkLibrary(deps.mlx_c_lib);
    exe.root_module.linkLibrary(deps.mlx_core_lib);
    exe.root_module.addSystemIncludePath(.{ .cwd_relative = "/opt/homebrew/include" });
    exe.root_module.linkFramework("Metal", .{});
    exe.root_module.linkFramework("Foundation", .{});
    exe.root_module.linkFramework("QuartzCore", .{});
    exe.root_module.linkFramework("Accelerate", .{});
    exe.root_module.linkSystemLibrary("c++", .{});
    exe.root_module.linkSystemLibrary("pcre2-8", .{});
}

fn configureLibModule(module: *std.Build.Module, deps: Dependencies) void {
    module.addSystemIncludePath(deps.mlx_c_dep.path("."));
    module.addSystemIncludePath(.{ .cwd_relative = "/opt/homebrew/include" });
}

fn addBuildMlxCoreStep(b: *std.Build, deps: Dependencies) void {
    const build_mlx_step = b.step("build-mlx-core", "Build core MLX static library with Zig");
    build_mlx_step.dependOn(&deps.mlx_core_lib.step);
}
