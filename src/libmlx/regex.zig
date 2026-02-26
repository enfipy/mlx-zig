//! regex.zig - Regex Bindings
//!
//! Copyright 2025 Joe

const std = @import("std");
const pcre2 = @cImport({
    @cDefine("PCRE2_CODE_UNIT_WIDTH", "8");
    @cInclude("pcre2.h");
});

pub const Regex = struct {
    const Self = @This();
    pattern: ?*pcre2.pcre2_code_8,
    match_data: ?*pcre2.pcre2_match_data_8,

    pub fn init(pattern_str: []const u8) !Self {
        var error_code: c_int = undefined;
        var error_offset: usize = undefined;
        const options = pcre2.PCRE2_UTF | pcre2.PCRE2_UCP;
        const pattern = pcre2.pcre2_compile_8(
            pattern_str.ptr,
            pattern_str.len,
            options,
            &error_code,
            &error_offset,
            null,
        );
        if (pattern == null) {
            var error_message: [256]u8 = undefined;
            _ = pcre2.pcre2_get_error_message_8(error_code, &error_message, error_message.len);
            std.debug.print(
                "PCRE2 compilation error at offset {d}: {s}\n",
                .{ error_offset, &error_message },
            );
            return error.RegexCompilationFailed;
        }
        const match_data = pcre2.pcre2_match_data_create_from_pattern_8(pattern, null);
        if (match_data == null) {
            pcre2.pcre2_code_free_8(pattern);
            return error.MatchDataCreationFailed;
        }
        return Self{
            .pattern = pattern,
            .match_data = match_data,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.match_data) |md| {
            pcre2.pcre2_match_data_free_8(md);
            self.match_data = null;
        }
        if (self.pattern) |p| {
            pcre2.pcre2_code_free_8(p);
            self.pattern = null;
        }
    }

    pub fn match(
        self: *Self,
        text: []const u8,
        start_pos: usize,
    ) !?struct { start: usize, end: usize } {
        const rc = pcre2.pcre2_match_8(
            self.pattern,
            text.ptr,
            text.len,
            start_pos,
            0,
            self.match_data,
            null,
        );
        if (rc < 0) {
            if (rc == pcre2.PCRE2_ERROR_NOMATCH) return null;
            return error.MatchingError;
        }
        const ovector = pcre2.pcre2_get_ovector_pointer_8(self.match_data);
        const match_start = ovector[0];
        const match_end = ovector[1];
        if (match_end <= match_start) return null;
        return .{ .start = match_start, .end = match_end };
    }
};
