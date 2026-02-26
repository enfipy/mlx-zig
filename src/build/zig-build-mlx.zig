const std = @import("std");

fn default_io() std.Io {
    return std.Io.Threaded.global_single_threaded.io();
}

// Build options equivalent to CMake options
const BuildOptions = struct {
    build_tests: bool,
    build_metal: bool,
    build_cpu: bool,
    enable_x64_mac: bool,
    build_gguf: bool,
    build_safetensors: bool,
    metal_jit: bool,
    metal_output_path: []const u8,

    fn fromOptions(b: *std.Build) !BuildOptions {
        const default_rel_dir = "lib/metal";
        const default_path = try std.fmt.allocPrint(b.allocator, "{s}/{s}", .{ b.install_prefix, default_rel_dir });
        const description = try std.fmt.allocPrint(b.allocator, "Absolute path to the metallib. Defaults to {s}", .{default_path});

        return .{
            .build_tests = b.option(bool, "build-tests", "Build tests for mlx") orelse true,
            .build_metal = b.option(bool, "build-metal", "Build metal backend") orelse true,
            .build_cpu = b.option(bool, "build-cpu", "Build cpu backend") orelse true,
            .enable_x64_mac = b.option(bool, "enable-x64-mac", "Enable building for x64 macOS") orelse false,
            .build_gguf = b.option(bool, "build-gguf", "Include support for GGUF format") orelse true,
            .build_safetensors = b.option(bool, "build-safetensors", "Include support for safetensors format") orelse true,
            .metal_jit = b.option(bool, "metal-jit", "Use JIT compilation for Metal kernels") orelse false,

            .metal_output_path = b.option([]const u8, "metal-output-path", description) orelse default_path,
        };
    }
};

const CPP_FLAGS = [_][]const u8{
    "-std=c++17",
    "-fPIC",
    "-DACCELERATE_NEW_LAPACK", // TODO this should be set conditionally
    "-D_GLIBCXX_USE_CXX11_ABI=1",
    "-U__ARM_FEATURE_BF16",
    "-fexceptions",
};

const C_FLAGS = [_][]const u8{
    "-fPIC",
    "-DACCELERATE_NEW_LAPACK", // TODO this should be set conditionally
    "-D_GLIBCXX_USE_CXX11_ABI=1",
    "-U__ARM_FEATURE_BF16",
    "-fexceptions",
};

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = std.builtin.OptimizeMode.ReleaseFast;

    const options = try BuildOptions.fromOptions(b);
    const deps = try Dependencies.init(b, options, target, optimize);

    const og_mlx = b.dependency("mlx", .{
        .target = target,
        .optimize = optimize,
    });

    const lib = b.addLibrary(.{
        .linkage = .static,
        .name = "mlx",
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
        }),
    });

    lib.root_module.addIncludePath(deps.fmt.path("include"));
    lib.root_module.addCMacro("FMT_HEADER_ONLY", "1");

    lib.installHeadersDirectory(og_mlx.path("."), ".", .{});
    lib.root_module.addIncludePath(og_mlx.path("."));
    lib.root_module.linkSystemLibrary("c++", .{});

    // Add core sources
    lib.root_module.addCSourceFiles(.{ .root = og_mlx.path("mlx"), .files = &core_sources, .flags = &CPP_FLAGS });
    lib.root_module.addCSourceFile(.{
        .file = og_mlx.path("mlx/io/load.cpp"),
        .flags = &CPP_FLAGS,
    });

    if (options.build_safetensors) {
        if (deps.json) |json_dep| {
            lib.root_module.addIncludePath(json_dep.path("single_include/nlohmann"));
        }

        lib.root_module.addCSourceFile(.{
            .file = og_mlx.path("mlx/io/safetensors.cpp"),
            .flags = &CPP_FLAGS,
        });
    } else {
        lib.root_module.addCSourceFile(.{
            .file = og_mlx.path("mlx/io/no_safetensors.cpp"),
            .flags = &CPP_FLAGS,
        });
    }

    if (options.build_gguf) {
        const gguf_dep = deps.gguflib.?;

        lib.root_module.addIncludePath(gguf_dep.path("."));

        const gguflib_lib = b.addLibrary(.{
            .linkage = .static,
            .name = "gguflib",
            .root_module = b.createModule(.{
                .target = target,
                .optimize = optimize,
            }),
        });

        const gguflib_sources = [_][]const u8{
            "fp16.c",
            "gguflib.c",
        };

        gguflib_lib.root_module.addCSourceFiles(.{
            .root = gguf_dep.path("."),
            .files = &gguflib_sources,
            .flags = &C_FLAGS,
        });

        lib.root_module.linkLibrary(gguflib_lib);

        const gguf_sources = [_][]const u8{
            "io/gguf.cpp",
            "io/gguf_quants.cpp",
        };
        lib.root_module.addCSourceFiles(.{
            .root = og_mlx.path("mlx"),
            .files = &gguf_sources,
            .flags = &C_FLAGS,
        });
    } else {
        lib.root_module.addCSourceFile(.{
            .file = og_mlx.path("mlx/io/no_gguf.cpp"),
            .flags = &CPP_FLAGS,
        });
    }

    lib.root_module.addCSourceFiles(.{
        .root = og_mlx.path("mlx"),
        .files = &distributed_sources,
        .flags = &CPP_FLAGS,
    });

    const is_darwin = target.result.os.tag.isDarwin();
    const is_arm = std.Target.Cpu.Arch.isAARCH64(target.result.cpu.arch);
    const is_x86_64 = std.Target.Cpu.Arch.isX86(target.result.cpu.arch);
    const is_ios = target.result.os.tag == .ios;

    // Validate system requirements
    if (is_darwin and is_x86_64 and !options.enable_x64_mac) {
        @panic("Building for x86_64 on macOS is not supported. If you are on an Apple silicon system, check the build documentation.");
    }

    // Check SDK version for Metal
    if (is_darwin and options.build_metal) {
        const sdk_version = try checkMacOSSDKVersion();
        if (sdk_version < 14.0) {
            @panic("MLX requires macOS SDK >= 14.0 to be built with -Dbuild-metal=true");
        }
    }

    // Metal support (Darwin only)
    if (options.build_metal) {
        const root = deps.metal_cpp.?.path(".");
        lib.root_module.addIncludePath(root);
        lib.installHeadersDirectory(root, ".", .{ .include_extensions = &.{".hpp"} });

        lib.root_module.addCSourceFiles(.{ .root = og_mlx.path("mlx"), .files = &metal_sources, .flags = &CPP_FLAGS });

        const formatted_metal_output_path = try std.fmt.allocPrint(b.allocator, "\"{s}/mlx.metallib\"", .{options.metal_output_path});
        std.log.info("METAL_PATH: {s}", .{formatted_metal_output_path});
        lib.root_module.addCMacro("METAL_PATH", formatted_metal_output_path);

        try buildAllKernels(b, lib, og_mlx, options);

        if (options.metal_jit) {
            lib.root_module.addCSourceFile(.{ .file = og_mlx.path("mlx/backend/metal/jit_kernels.cpp") });
        } else {
            lib.root_module.addCSourceFile(.{ .file = og_mlx.path("mlx/backend/metal/nojit_kernels.cpp") });
        }

        try build_jit_sources(b, lib, og_mlx, options);

        lib.root_module.linkFramework("Metal", .{});
        lib.root_module.linkFramework("Foundation", .{});
        lib.root_module.linkFramework("QuartzCore", .{});
    } else {
        lib.root_module.addCSourceFiles(.{ .root = og_mlx.path("mlx"), .files = &no_metal_sources, .flags = &CPP_FLAGS });
    }

    if (options.build_cpu) {
        lib.root_module.addCSourceFiles(.{
            .root = og_mlx.path("mlx"),
            .files = &common_sources,
            .flags = &CPP_FLAGS,
        });

        if (is_ios) {
            lib.root_module.addCSourceFile(.{
                .file = og_mlx.path("mlx/backend/no_cpu/compiled.cpp"),
                .flags = &CPP_FLAGS,
            });
        } else {
            try build_preamble(b, lib, og_mlx, is_darwin);
            lib.root_module.addCSourceFiles(.{
                .root = og_mlx.path("mlx"),
                .files = &cpu_sources,
                .flags = &CPP_FLAGS,
            });

            if (is_darwin and is_arm) {
                lib.root_module.linkFramework("Accelerate", .{});
                lib.root_module.addCMacro("MLX_USE_ACCELERATE", "1");
                lib.root_module.addCMacro("ACCELERATE_NEW_LAPACK", "1");
                lib.root_module.addCSourceFiles(.{
                    .root = og_mlx.path("mlx"),
                    .files = &cpu_gemm_accelerate_sources,
                    .flags = &CPP_FLAGS,
                });
            } else {
                lib.root_module.addCSourceFiles(.{
                    .root = og_mlx.path("mlx"),
                    .files = &cpu_gemm_fallback_sources,
                    .flags = &CPP_FLAGS,
                });

                if (!is_darwin) {
                    lib.root_module.linkSystemLibrary("lapack", .{});
                    lib.root_module.linkSystemLibrary("blas", .{});
                }
            }
        }
    } else {
        lib.root_module.addCSourceFiles(.{ .root = og_mlx.path("mlx"), .files = &no_cpu_sources, .flags = &CPP_FLAGS });
    }

    if (!is_darwin) {
        lib.root_module.linkSystemLibrary("stdc++", .{});
    }

    b.installArtifact(lib);

    if (options.build_tests) {
        const tests = b.addExecutable(.{
            .name = "tests",
            .root_module = b.createModule(.{
                .target = target,
                .optimize = optimize,
            }),
        });

        tests.root_module.addIncludePath(deps.doctest.?.path("."));
        tests.root_module.linkLibrary(lib);

        tests.root_module.addCSourceFiles(.{ .root = og_mlx.path("."), .files = &test_sources, .flags = &CPP_FLAGS });

        if (options.build_metal) {
            tests.root_module.addCSourceFile(.{ .file = og_mlx.path("tests/metal_tests.cpp"), .flags = &CPP_FLAGS });
        }

        const test_step = b.step("test", "Run library tests");
        const run_cmd = b.addRunArtifact(tests);
        test_step.dependOn(&run_cmd.step);

        b.installArtifact(tests);
    }
}

/////////////////////////////////////////
/// Everything to build metal kernels
///////////////////////////////////////

fn buildAllKernels(b: *std.Build, lib: *std.Build.Step.Compile, og_mlx: *std.Build.Dependency, options: BuildOptions) !void {
    var airFiles = std.array_list.Managed(std.Build.LazyPath).init(b.allocator);
    defer airFiles.deinit();

    // always build
    inline for (default_kernels) |kernel| {
        try airFiles.append(try buildKernel(b, kernel, og_mlx));
    }

    if (!options.metal_jit) {
        inline for (jit_kernels) |kernel| {
            try airFiles.append(try buildKernel(b, kernel, og_mlx));
        }
    }

    // finally build the metallib which depends on all the air files
    try buildMetallib(b, lib, airFiles.items, options);
}

fn buildKernel(b: *std.Build, comptime rel_path: []const u8, og_mlx: *std.Build.Dependency) !std.Build.LazyPath {
    const name = comptime (if (std.mem.lastIndexOf(u8, rel_path, "/")) |last_slash|
        rel_path[(last_slash + 1)..]
    else
        rel_path);

    var metal_flags = std.array_list.Managed([]const u8).init(b.allocator);
    defer metal_flags.deinit();

    try metal_flags.appendSlice(&[_][]const u8{
        "-Wall",
        "-Wextra",
        "-fno-fast-math",
    });

    const version_include = getVersionIncludes(310);
    try metal_flags.appendSlice(&[_][]const u8{
        "-I",
        og_mlx.path(version_include).getPath(b),
    });

    // In the CMake PROJECT_SOURCE_DIR is always included
    try metal_flags.appendSlice(&[_][]const u8{
        "-I",
        og_mlx.path(".").getPath(b),
    });

    const source_path = "mlx/backend/metal/kernels/" ++ rel_path ++ ".metal";
    const source_path_lazy = og_mlx.path(source_path);

    const metal_cmd = b.addSystemCommand(&[_][]const u8{
        "xcrun",
        "-sdk",
        "macosx",
        "metal",
    });
    metal_cmd.addArgs(metal_flags.items);

    metal_cmd.addArg("-c");
    metal_cmd.addArg(source_path_lazy.getPath(b));
    metal_cmd.addArg("-o");
    const out_file_name = name ++ ".air";
    const output_path = metal_cmd.addOutputFileArg(out_file_name);

    const dest_rel_path = "include/mlx/backend/metal/kernels/" ++ name ++ ".air";
    const metal_install = b.addInstallFile(output_path, dest_rel_path);
    metal_install.step.dependOn(&metal_cmd.step);

    b.default_step.dependOn(&metal_install.step);

    return output_path;
}

fn buildMetallib(b: *std.Build, lib: *std.Build.Step.Compile, air_files: []std.Build.LazyPath, options: BuildOptions) !void {
    const metallib_cmd = b.addSystemCommand(&[_][]const u8{
        "xcrun",
        "-sdk",
        "macosx",
        "metallib",
    });

    for (air_files) |air| {
        metallib_cmd.addFileArg(air);
    }

    metallib_cmd.addArg("-o");
    const metallib_file = metallib_cmd.addOutputFileArg("mlx.metallib");

    const copy_step = CopyMetalLibStep.create(b, metallib_file, options.metal_output_path);

    copy_step.step.dependOn(&metallib_cmd.step);
    lib.step.dependOn(&copy_step.step);
}

const CopyMetalLibStep = struct {
    const Self = @This();

    step: std.Build.Step,
    b: *std.Build,
    metallib_file: std.Build.LazyPath,
    metal_output_path: []const u8,

    pub fn create(b: *std.Build, metallib_file: std.Build.LazyPath, metal_output_path: []const u8) *CopyMetalLibStep {
        const new = b.allocator.create(Self) catch unreachable;
        new.* = .{
            .b = b,
            .metallib_file = metallib_file,
            .metal_output_path = metal_output_path,
            .step = std.Build.Step.init(.{
                .id = .custom,
                .name = "copy_mlx_metallib",
                .owner = b,
                .makeFn = make,
            }),
        };
        return new;
    }

    fn make(step: *std.Build.Step, make_options: std.Build.Step.MakeOptions) anyerror!void {
        _ = make_options;

        const self: *Self = @fieldParentPtr("step", step);
        const src_path = self.metallib_file.getPath(self.b);

        const dest_file = try std.fmt.allocPrint(self.b.allocator, "{s}/mlx.metallib", .{self.metal_output_path});

        const io = default_io();
        _ = try std.Io.Dir.cwd().createDirPathStatus(io, self.metal_output_path, .default_dir);

        try std.Io.Dir.copyFileAbsolute(src_path, dest_file, io, .{});
    }
};

fn build_jit_sources(b: *std.Build, lib: *std.Build.Step.Compile, og_mlx: *std.Build.Dependency, options: BuildOptions) !void {
    inline for (default_jit_sources) |source| {
        try make_jit_source(b, lib, og_mlx, source);
    }

    if (options.metal_jit) {
        inline for (optional_jit_sources) |source| {
            try make_jit_source(b, lib, og_mlx, source);
        }
    }
}

fn make_jit_source(b: *std.Build, lib: *std.Build.Step.Compile, og_mlx: *std.Build.Dependency, comptime name: []const u8) !void {
    const wf = b.addWriteFiles();

    const header_file_name = name ++ ".h";
    const source_dir = b.pathJoin(&.{og_mlx.path(".").getPath(b)});
    const jit_includes = og_mlx.path("mlx/backend/metal/kernels/jit").getPath(b);
    const headerPath = b.pathJoin(&.{ og_mlx.path("mlx/backend/metal/kernels").getPath(b), header_file_name });

    // kinda rouge to discard the errros but this is also done in the original MLX repo
    const commandStr = try std.fmt.allocPrint(b.allocator, "{s} -I{s} -I{s} -DMLX_METAL_JIT -E -P {s} 2>/dev/null || true", .{
        // TODO don't hard code this cc compiler and fix clang++: warning: treating 'c-header' input as 'c++-header' when in C++ mode, this behavior is deprecated [-Wdeprecated]
        "c++",
        source_dir,
        jit_includes,
        headerPath,
    });
    defer b.allocator.free(commandStr);

    const preprocess = b.addSystemCommand(&[_][]const u8{
        "sh",
        "-c",
        commandStr,
    });

    const std_out_path = preprocess.captureStdOut(.{});

    const read_step = ReadFileStep.create(b, std_out_path);
    read_step.step.dependOn(&preprocess.step);

    const gen_step = GeneratePreambleStep.create(name, b, read_step, false, wf, lib, MetalOrCommon.metal);
    gen_step.step.dependOn(&read_step.step);

    wf.step.dependOn(&gen_step.step);

    const add_step = AddFileStep.create(b, lib, gen_step);
    add_step.step.dependOn(&wf.step);

    lib.step.dependOn(&add_step.step);
}

/////////////////////////////////////////
/// Build Preamble
///////////////////////////////////////

fn build_preamble(b: *std.Build, lib: *std.Build.Step.Compile, og_mlx: *std.Build.Dependency, is_darwin: bool) !void {
    const wf = b.addWriteFiles();

    const preprocess = b.addSystemCommand(&[_][]const u8{
        "c++",
        "-I",
        b.pathJoin(&.{og_mlx.path(".").getPath(b)}),
        "-E",
        b.pathJoin(&.{ og_mlx.path("mlx").getPath(b), "backend", "cpu", "compiled_preamble.h" }),
    });

    const std_out_path = preprocess.captureStdOut(.{});

    const read_step = ReadFileStep.create(b, std_out_path);
    read_step.step.dependOn(&preprocess.step);

    const gen_step = GeneratePreambleStep.create("compiled_preamble", b, read_step, is_darwin, wf, lib, MetalOrCommon.common);
    gen_step.step.dependOn(&read_step.step);

    wf.step.dependOn(&gen_step.step);

    const add_step = AddFileStep.create(b, lib, gen_step);
    add_step.step.dependOn(&wf.step);

    lib.step.dependOn(&add_step.step);
}

/////////////////////////////////////////
/// Custom Steps
///////////////////////////////////////

const ReadFileStep = struct {
    const Self = @This();

    step: std.Build.Step,
    b: *std.Build,
    path: std.Build.LazyPath,
    contents: []const u8 = "",

    fn create(b: *std.Build, path: std.Build.LazyPath) *Self {
        const new = b.allocator.create(Self) catch unreachable;
        new.* = .{
            .step = std.Build.Step.init(.{
                .id = .custom,
                .name = "read_file",
                .owner = b,
                .makeFn = make,
            }),
            .path = path,
            .b = b,
        };
        return new;
    }

    fn make(step: *std.Build.Step, make_options: std.Build.Step.MakeOptions) anyerror!void {
        _ = make_options;
        const self: *Self = @fieldParentPtr("step", step);
        const path = self.path.getPath(self.b);

        const io = default_io();
        self.contents = try std.Io.Dir.cwd().readFileAlloc(
            io,
            path,
            self.b.allocator,
            .limited(std.math.maxInt(usize)),
        );
    }
};

const MetalOrCommon = enum { common, metal };

const GeneratePreambleStep = struct {
    const Self = @This();

    name: []const u8,
    step: std.Build.Step,
    b: *std.Build,
    read_step: *ReadFileStep,
    is_darwin: bool,
    wf: *std.Build.Step.WriteFile,
    output_path: std.Build.LazyPath,
    lib: *std.Build.Step.Compile,
    which: MetalOrCommon,

    pub fn create(
        name: []const u8,
        b: *std.Build,
        read_step: *ReadFileStep,
        is_darwin: bool,
        wf: *std.Build.Step.WriteFile,
        lib: *std.Build.Step.Compile,
        which: MetalOrCommon,
    ) *GeneratePreambleStep {
        const new = b.allocator.create(Self) catch unreachable;

        new.* = .{
            .name = name,
            .step = std.Build.Step.init(.{
                .id = .custom,
                .name = "generate_preamble",
                .owner = b,
                .makeFn = make,
            }),
            .b = b,
            .read_step = read_step,
            .is_darwin = is_darwin,
            .wf = wf,
            .output_path = undefined,
            .lib = lib,
            .which = which,
        };

        return new;
    }

    fn make(step: *std.Build.Step, make_options: std.Build.Step.MakeOptions) anyerror!void {
        _ = make_options;
        const self: *Self = @fieldParentPtr("step", step);

        switch (self.which) {
            MetalOrCommon.common => try make_common_step(self),
            MetalOrCommon.metal => try make_jit_step(self),
        }
    }

    fn make_common_step(self: *GeneratePreambleStep) anyerror!void {
        var content = std.array_list.Managed(u8).init(self.b.allocator);
        defer content.deinit();

        try content.appendSlice(
            \\const char* get_kernel_preamble() {
            \\return R"preamble(
            \\
        );

        if (self.is_darwin) {
            try content.appendSlice(
                \\#include <cmath>
                \\#include <complex>
                \\#include <cstdint>
                \\#include <vector>
                \\
            );
        }

        try content.appendSlice(self.read_step.contents);

        try content.appendSlice(
            \\
            \\using namespace mlx::core;
            \\using namespace mlx::core::detail;
            \\)preamble";
            \\}
            \\
        );

        self.output_path = self.wf.add("mlx/backend/cpu/compiled_preamble.cpp", content.items);
    }

    fn make_jit_step(self: *GeneratePreambleStep) anyerror!void {
        var content = std.array_list.Managed(u8).init(self.b.allocator);
        defer content.deinit();

        try content.appendSlice(
            \\namespace mlx::core::metal {
            \\
            \\
        );

        const line = try std.fmt.allocPrint(
            self.b.allocator,
            "const char* {s} () {{\nreturn R\"preamble(\n",
            .{self.name},
        );

        try content.appendSlice(line);

        try content.appendSlice(self.read_step.contents);

        try content.appendSlice(
            \\)preamble";
            \\}
            \\
            \\} // namespace mlx::core::metal
            \\
        );

        const output_path = try std.fmt.allocPrint(self.b.allocator, "mlx/backend/metal/jit/{s}.cpp", .{self.name});
        self.output_path = self.wf.add(output_path, content.items);
    }
};

const AddFileStep = struct {
    const Self = @This();

    step: std.Build.Step,
    b: *std.Build,
    gen_step: *GeneratePreambleStep,
    lib: *std.Build.Step.Compile,

    fn create(b: *std.Build, lib: *std.Build.Step.Compile, gen_step: *GeneratePreambleStep) *Self {
        const new = b.allocator.create(Self) catch unreachable;
        new.* = .{
            .step = std.Build.Step.init(.{
                .id = .custom,
                .name = "add_file",
                .owner = b,
                .makeFn = make,
            }),
            .b = b,
            .lib = lib,
            .gen_step = gen_step,
        };
        return new;
    }

    fn make(step: *std.Build.Step, make_options: std.Build.Step.MakeOptions) anyerror!void {
        _ = make_options;
        const self: *Self = @fieldParentPtr("step", step);

        const preamble_cpp_file = self.gen_step.output_path;
        self.lib.root_module.addCSourceFile(.{ .file = preamble_cpp_file, .flags = &CPP_FLAGS });
    }
};

///////////////////////////////////////////
/// Build deps like gguf, safetensors etc.
//////////////////////////////////////////

const Dependencies = struct {
    fmt: *std.Build.Dependency,
    doctest: ?*std.Build.Dependency,
    json: ?*std.Build.Dependency,
    gguflib: ?*std.Build.Dependency,
    metal_cpp: ?*std.Build.Dependency = null,

    fn init(b: *std.Build, options: BuildOptions, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) !Dependencies {
        const fmt = b.dependency("fmt", .{
            .target = target,
            .optimize = optimize,
        });

        const doctest = if (options.build_tests) b.dependency("doctest", .{
            .target = target,
            .optimize = optimize,
        }) else null;

        const json = if (options.build_safetensors) b.dependency("json", .{
            .target = target,
            .optimize = optimize,
        }) else null;

        const gguflib = if (options.build_gguf) b.dependency("gguflib", .{
            .target = target,
            .optimize = optimize,
        }) else null;

        const metal_cpp = if (options.build_metal) b.dependency("metal-cpp", .{
            .target = target,
            .optimize = optimize,
        }) else null;

        return Dependencies{
            .fmt = fmt,
            .doctest = doctest,
            .json = json,
            .gguflib = gguflib,
            .metal_cpp = metal_cpp,
        };
    }
};

////////////////////////////
/// util functions
////////////////////////////

fn checkMacOSSDKVersion() !f32 {
    const io = default_io();
    const result = try std.process.run(std.heap.page_allocator, io, .{
        .argv = &[_][]const u8{ "xcrun", "-sdk", "macosx", "--show-sdk-version" },
    });
    defer std.heap.page_allocator.free(result.stdout);
    defer std.heap.page_allocator.free(result.stderr);

    const version = try std.fmt.parseFloat(f32, std.mem.trim(u8, result.stdout, " \n\r"));
    return version;
}

fn getVersionIncludes(metal_version: u32) []const u8 {
    return if (metal_version >= 310)
        "mlx/backend/metal/kernels/metal_3_1"
    else
        "mlx/backend/metal/kernels/metal_3_0";
}

/////////////////////////
/// Files
/////////////////////////

const core_sources = [_][]const u8{
    "allocator.cpp",
    "array.cpp",
    "compile.cpp",
    "device.cpp",
    "dtype.cpp",
    "export.cpp",
    "einsum.cpp",
    "fast.cpp",
    "fft.cpp",
    "ops.cpp",
    "graph_utils.cpp",
    "primitives.cpp",
    "random.cpp",
    "scheduler.cpp",
    "transforms.cpp",
    "utils.cpp",
    "linalg.cpp",
};

const distributed_sources = [_][]const u8{
    "distributed/primitives.cpp",
    "distributed/ops.cpp",
    "distributed/distributed.cpp",
    "distributed/mpi/no_mpi.cpp",
    "distributed/ring/no_ring.cpp",
};

const metal_sources = [_][]const u8{
    "backend/metal/allocator.cpp",
    "backend/metal/binary.cpp",
    "backend/metal/conv.cpp",
    "backend/metal/compiled.cpp",
    "backend/metal/copy.cpp",
    "backend/metal/custom_kernel.cpp",
    "backend/metal/distributed.cpp",
    "backend/metal/device.cpp",
    "backend/metal/event.cpp",
    "backend/metal/fence.cpp",
    "backend/metal/fft.cpp",
    "backend/metal/hadamard.cpp",
    "backend/metal/indexing.cpp",
    "backend/metal/matmul.cpp",
    "backend/metal/scaled_dot_product_attention.cpp",
    "backend/metal/metal.cpp",
    "backend/metal/primitives.cpp",
    "backend/metal/quantized.cpp",
    "backend/metal/normalization.cpp",
    "backend/metal/rope.cpp",
    "backend/metal/scan.cpp",
    "backend/metal/slicing.cpp",
    "backend/metal/softmax.cpp",
    "backend/metal/sort.cpp",
    "backend/metal/reduce.cpp",
    "backend/metal/ternary.cpp",
    "backend/metal/unary.cpp",
    "backend/metal/resident.cpp",
    "backend/metal/utils.cpp",
};

const default_kernels = [_][]const u8{
    "arg_reduce",
    "conv",
    "gemv",
    "layer_norm",
    "random",
    "rms_norm",
    "rope",
    "scaled_dot_product_attention",
    "steel/attn/kernels/steel_attention",
};

const jit_kernels = [_][]const u8{
    "arange",
    "binary",
    "binary_two",
    "copy",
    "fft",
    "reduce",
    "quantized",
    "scan",
    "softmax",
    "sort",
    "ternary",
    "unary",
    "steel/conv/kernels/steel_conv",
    "steel/conv/kernels/steel_conv_general",
    "steel/gemm/kernels/steel_gemm_fused",
    "steel/gemm/kernels/steel_gemm_masked",
    "steel/gemm/kernels/steel_gemm_splitk",
    "gemv_masked",
};

const default_jit_sources = [_][]const u8{
    "utils",
    "unary_ops",
    "binary_ops",
    "ternary_ops",
    "reduce_utils",
    "scatter",
    "gather",
    "gather_axis",
    "scatter_axis",
    "hadamard",
};

const optional_jit_sources = [_][]const u8{
    "arange",
    "copy",
    "unary",
    "binary",
    "binary_two",
    "fft",
    "ternary",
    "softmax",
    "scan",
    "sort",
    "reduce",
    "steel/gemm/gemm",
    "steel/gemm/kernels/steel_gemm_fused",
    "steel/gemm/kernels/steel_gemm_masked",
    "steel/gemm/kernels/steel_gemm_splitk",
    "steel/conv/conv",
    "steel/conv/kernels/steel_conv",
    "steel/conv/kernels/steel_conv_general",
    "quantized",
    "gemv_masked",
};

const no_metal_sources = [_][]const u8{
    "backend/no_metal/allocator.cpp",
    "backend/no_metal/event.cpp",
    "backend/no_metal/metal.cpp",
    "backend/no_metal/primitives.cpp",
};

const common_sources = [_][]const u8{
    "backend/common/compiled.cpp",
    "backend/common/common.cpp",
    "backend/common/load.cpp",
    "backend/common/reduce.cpp",
    "backend/common/slicing.cpp",
    "backend/common/utils.cpp",
};

const cpu_sources = [_][]const u8{
    "backend/cpu/arg_reduce.cpp",
    "backend/cpu/binary.cpp",
    "backend/cpu/conv.cpp",
    "backend/cpu/copy.cpp",
    "backend/cpu/eigh.cpp",
    "backend/cpu/fft.cpp",
    "backend/cpu/hadamard.cpp",
    "backend/cpu/matmul.cpp",
    "backend/cpu/gemms/cblas.cpp",
    "backend/cpu/masked_mm.cpp",
    "backend/cpu/primitives.cpp",
    "backend/cpu/quantized.cpp",
    "backend/cpu/reduce.cpp",
    "backend/cpu/scan.cpp",
    "backend/cpu/select.cpp",
    "backend/cpu/softmax.cpp",
    "backend/cpu/sort.cpp",
    "backend/cpu/threefry.cpp",
    "backend/cpu/indexing.cpp",
    "backend/cpu/luf.cpp",
    "backend/cpu/qrf.cpp",
    "backend/cpu/svd.cpp",
    "backend/cpu/inverse.cpp",
    "backend/cpu/cholesky.cpp",
    "backend/cpu/unary.cpp",
    "backend/cpu/compiled.cpp",
    "backend/cpu/jit_compiler.cpp",
};

const cpu_gemm_accelerate_sources = [_][]const u8{
    "backend/cpu/gemms/bnns.cpp",
};

const cpu_gemm_fallback_sources = [_][]const u8{
    "backend/cpu/gemms/no_fp16.cpp",
    "backend/cpu/gemms/no_bf16.cpp",
};

const no_cpu_sources = [_][]const u8{
    "backend/no_cpu/primitives.cpp",
    "backend/no_cpu/compiled.cpp",
};

const test_sources = [_][]const u8{
    "tests/tests.cpp",
    "tests/allocator_tests.cpp",
    "tests/array_tests.cpp",
    "tests/arg_reduce_tests.cpp",
    "tests/autograd_tests.cpp",
    "tests/blas_tests.cpp",
    "tests/compile_tests.cpp",
    "tests/custom_vjp_tests.cpp",
    "tests/creations_tests.cpp",
    "tests/device_tests.cpp",
    "tests/einsum_tests.cpp",
    "tests/eval_tests.cpp",
    "tests/fft_tests.cpp",
    "tests/load_tests.cpp",
    "tests/ops_tests.cpp",
    "tests/random_tests.cpp",
    "tests/scheduler_tests.cpp",
    "tests/utils_tests.cpp",
    "tests/vmap_tests.cpp",
    "tests/linalg_tests.cpp",
};
