const std = @import("std");
const impl = @import("zig-build-mlx.zig");

pub fn build(b: *std.Build) !void {
    try impl.build(b);
}
