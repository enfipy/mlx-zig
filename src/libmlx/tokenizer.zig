//! tokenizer.zig - TikToken and Huggingface Tokenizer
//!
//! Copyright 2025 Joe

const std = @import("std");
const Regex = @import("regex.zig").Regex;
const format_dynamic = @import("utils.zig").format_dynamic;
const TextPart = @import("utils.zig").TextPart;

fn default_io() std.Io {
    return std.Io.Threaded.global_single_threaded.io();
}

const default_model_assets_dir = "src/assets/models";
const tokenizer_model_file = "tokenizer.model";
const model_assets_env_var = "MLX_MODEL_ASSETS_DIR";

fn model_path_for_test(allocator: std.mem.Allocator) ![]u8 {
    const model_assets_dir = std.process.getEnvVarOwned(
        allocator,
        model_assets_env_var,
    ) catch |err| {
        if (err == error.EnvironmentVariableNotFound) {
            return std.fs.path.join(
                allocator,
                &.{ default_model_assets_dir, tokenizer_model_file },
            );
        }
        return err;
    };
    defer allocator.free(model_assets_dir);

    if (model_assets_dir.len == 0) {
        return std.fs.path.join(
            allocator,
            &.{ default_model_assets_dir, tokenizer_model_file },
        );
    }

    return std.fs.path.join(
        allocator,
        &.{ model_assets_dir, tokenizer_model_file },
    );
}

pub const Tokenizer = struct {
    const Self = @This();
    allocator: std.mem.Allocator,
    pattern_regex: ?Regex,
    special_regex: ?Regex,
    vocab: std.StringHashMap(u32),
    id_to_token: std.AutoHashMap(u32, []const u8),
    specials: []const []const u8,

    pub fn init(allocator: std.mem.Allocator, path_json: []const u8) !Self {
        const json_path = try std.fmt.allocPrint(allocator, "{s}/tokenizer.json", .{path_json});
        defer allocator.free(json_path);
        const io = default_io();
        const json_content = try std.Io.Dir.cwd().readFileAlloc(
            io,
            json_path,
            allocator,
            .limited(100 * 1024 * 1024),
        );
        defer allocator.free(json_content);
        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
        defer parsed.deinit();
        var pattern_regex: ?Regex = null;
        const pre_tokenizer = parsed.value.object.get("pre_tokenizer");
        if (pre_tokenizer != null) {
            const pretokenizers = pre_tokenizer.?.object.get("pretokenizers");
            if (pretokenizers != null and pretokenizers.?.array.items.len > 0) {
                const pattern_obj = pretokenizers.?.array.items[0].object.get("pattern");
                if (pattern_obj != null and pattern_obj.?.object.get("Regex") != null) {
                    const pattern = pattern_obj.?.object.get("Regex").?.string;
                    pattern_regex = try Regex.init(pattern);
                }
            }
        }

        var vocab = std.StringHashMap(u32).init(allocator);
        errdefer vocab.deinit();
        var id_to_token = std.AutoHashMap(u32, []const u8).init(allocator);
        errdefer id_to_token.deinit();
        const replacements = [_]struct { bytes: [2]u8, replacement: u8 }{
            .{ .bytes = .{ 0xC4, 0xA0 }, .replacement = ' ' },
            .{ .bytes = .{ 0xC4, 0x8A }, .replacement = '\n' },
            .{ .bytes = .{ 0xC4, 0x89 }, .replacement = '\t' },
        };
        var vocab_iter = parsed.value.object.get("model").?.object.get("vocab").?.object.iterator();
        while (vocab_iter.next()) |entry| {
            const token_str = entry.key_ptr.*;
            const id = @as(u32, @intCast(entry.value_ptr.*.integer));
            var buf = std.array_list.Managed(u8).init(allocator);
            defer buf.deinit();
            var i: usize = 0;
            while (i < token_str.len) {
                var replaced = false;
                for (replacements) |repl| {
                    if (i + 1 < token_str.len and
                        token_str[i] == repl.bytes[0] and
                        token_str[i + 1] == repl.bytes[1])
                    {
                        try buf.append(repl.replacement);
                        i += 2;
                        replaced = true;
                        break;
                    }
                }
                if (!replaced) {
                    try buf.append(token_str[i]);
                    i += 1;
                }
            }
            const processed_token = try buf.toOwnedSlice();
            try vocab.put(processed_token, id);
            try id_to_token.put(id, processed_token);
        }
        var special_tokens = std.array_list.Managed([]const u8).init(allocator);
        defer special_tokens.deinit();
        const added_tokens = parsed.value.object.get("added_tokens").?.array;
        for (added_tokens.items) |token| {
            if (token != .object) continue;
            const content = token.object.get("content") orelse continue;
            const id = token.object.get("id") orelse continue;
            if (content != .string or id != .integer) continue;
            const token_id = @as(u32, @intCast(id.integer));
            const token_str = content.string;
            if (vocab.get(token_str)) |old_id| {
                if (old_id != token_id) {
                    std.debug.print(
                        "Duplicate: {s} {d} -> {d}",
                        .{ token_str, old_id, token_id },
                    );
                }
                continue;
            }
            const token_copy = try allocator.dupe(u8, token_str);
            try vocab.put(token_copy, token_id);
            try id_to_token.put(token_id, token_copy);
            try special_tokens.append(token_copy);
        }
        const specials = try allocator.dupe([]const u8, special_tokens.items);
        const special_regex = try create_special_regex(allocator, specials);
        return Self{
            .allocator = allocator,
            .pattern_regex = pattern_regex,
            .special_regex = special_regex,
            .vocab = vocab,
            .id_to_token = id_to_token,
            .specials = specials,
        };
    }

    pub fn init_from_tik_token(
        allocator: std.mem.Allocator,
        pattern: []const u8,
        vocabulary_path: []const u8,
        specials_: []const []const u8,
    ) !Self {
        const specials = try allocator.dupe([]const u8, specials_);
        var pattern_regex = try Regex.init(pattern);
        errdefer pattern_regex.deinit();
        var vocab = std.StringHashMap(u32).init(allocator);
        errdefer vocab.deinit();
        var id_to_token = std.AutoHashMap(u32, []const u8).init(allocator);
        errdefer id_to_token.deinit();
        const io = default_io();
        const file_content = try std.Io.Dir.cwd().readFileAlloc(
            io,
            vocabulary_path,
            allocator,
            .limited(256 * 1024 * 1024),
        );
        defer allocator.free(file_content);
        var lines = std.mem.splitScalar(u8, file_content, '\n');
        while (lines.next()) |line| {
            if (line.len == 0) continue;
            var iter = std.mem.tokenizeAny(u8, line, " ");
            const token_b64 = iter.next() orelse continue;
            const rank_str = iter.next() orelse continue;
            if (std.mem.eql(u8, token_b64, "=")) {
                const token = try allocator.alloc(u8, 0);
                const rank = try std.fmt.parseInt(u32, rank_str, 10);
                try vocab.put(token, rank);
                try id_to_token.put(rank, token);
                continue;
            }
            const decoded_size = try std.base64.standard.Decoder.calcSizeForSlice(token_b64);
            const token = try allocator.alloc(u8, decoded_size);
            errdefer allocator.free(token);
            _ = try std.base64.standard.Decoder.decode(token, token_b64);
            const rank = try std.fmt.parseInt(u32, rank_str, 10);
            try vocab.put(token, rank);
            try id_to_token.put(rank, token);
        }
        const special_start = @as(u32, @intCast(vocab.count()));
        for (specials, 0..) |special, i| {
            const token = try allocator.dupe(u8, special);
            const rank = special_start + @as(u32, @intCast(i));
            try vocab.put(token, rank);
            try id_to_token.put(rank, token);
        }
        const special_regex = try create_special_regex(allocator, specials);
        return Self{
            .allocator = allocator,
            .pattern_regex = pattern_regex,
            .special_regex = special_regex,
            .vocab = vocab,
            .id_to_token = id_to_token,
            .specials = specials,
        };
    }

    pub fn deinit(self: *Self) void {
        var it = self.vocab.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.vocab.deinit();
        self.id_to_token.deinit();
        if (self.special_regex) |*regex| {
            regex.deinit();
        }
        if (self.pattern_regex) |*regex| {
            regex.deinit();
        }
        self.allocator.free(self.specials);
    }

    fn load_vocabulary(self: *Self, path: []const u8) !void {
        const io = default_io();
        const file_content = try std.Io.Dir.cwd().readFileAlloc(
            io,
            path,
            self.allocator,
            .limited(256 * 1024 * 1024),
        );
        defer self.allocator.free(file_content);
        var lines = std.mem.splitScalar(u8, file_content, '\n');
        while (lines.next()) |line| {
            if (line.len == 0) continue;
            var iter = std.mem.tokenizeAny(u8, line, " ");
            const token_b64 = iter.next() orelse continue;
            const rank_str = iter.next() orelse continue;
            const decoded_size = try std.base64.standard.Decoder.calcSizeForSlice(token_b64);
            const token = try self.allocator.alloc(u8, decoded_size);
            errdefer self.allocator.free(token);
            _ = try std.base64.standard.Decoder.decode(token, token_b64);
            const rank = try std.fmt.parseInt(u32, rank_str, 10);
            try self.vocab.put(token, rank);
            try self.id_to_token.put(rank, token);
        }
        const special_start = @as(u32, @intCast(self.vocab.count()));
        for (self.specials, 0..) |special, i| {
            const token = try self.allocator.dupe(u8, special);
            errdefer self.allocator.free(token);
            const rank = special_start + @as(u32, @intCast(i));
            try self.vocab.put(token, rank);
            try self.id_to_token.put(rank, token);
        }
    }

    fn create_special_regex(allocator: std.mem.Allocator, specials: []const []const u8) !?Regex {
        if (specials.len == 0) return null;
        var pattern = std.array_list.Managed(u8).init(allocator);
        defer pattern.deinit();
        var first = true;
        for (specials) |special| {
            if (!first) {
                try pattern.append('|');
            } else {
                first = false;
            }
            for (special) |char| {
                if (std.mem.indexOfScalar(u8, "\\^$.|?*+()[{", char) != null) {
                    try pattern.append('\\');
                }
                try pattern.append(char);
            }
        }
        if (pattern.items.len == 0) return null;
        return try Regex.init(pattern.items);
    }

    fn split_with_specials(self: *Self, text: []const u8) !std.array_list.Managed([]const u8) {
        var result = std.array_list.Managed([]const u8).init(self.allocator);
        errdefer {
            for (result.items) |item| {
                self.allocator.free(item);
            }
            result.deinit();
        }
        if (self.special_regex == null) {
            try result.append(try self.allocator.dupe(u8, text));
            return result;
        }
        var last_end: usize = 0;
        var start: usize = 0;
        while (start < text.len) {
            const match_result = try self.special_regex.?.match(text, start);
            if (match_result == null) break;
            const match = match_result.?;
            if (match.start > last_end) {
                const non_match = try self.allocator.dupe(u8, text[last_end..match.start]);
                try result.append(non_match);
            }
            const matched_special = try self.allocator.dupe(u8, text[match.start..match.end]);
            try result.append(matched_special);
            last_end = match.end;
            start = match.end;
        }
        if (last_end < text.len) {
            const remaining = try self.allocator.dupe(u8, text[last_end..]);
            try result.append(remaining);
        }
        return result;
    }

    fn split_with_pattern(self: *Self, text: []const u8) !std.array_list.Managed([]const u8) {
        var result = std.array_list.Managed([]const u8).init(self.allocator);
        errdefer {
            for (result.items) |item| {
                self.allocator.free(item);
            }
            result.deinit();
        }
        if (self.pattern_regex == null) {
            try result.append(try self.allocator.dupe(u8, text));
            return result;
        }
        var start: usize = 0;
        while (start < text.len) {
            const match_result = try self.pattern_regex.?.match(text, start);
            if (match_result == null) {
                if (start < text.len) {
                    const remaining = try self.allocator.dupe(u8, text[start..]);
                    try result.append(remaining);
                }
                break;
            }
            const match = match_result.?;
            const matched_text = try self.allocator.dupe(u8, text[match.start..match.end]);
            try result.append(matched_text);
            start = match.end;
        }
        return result;
    }

    fn bpe_merges(self: *Self, token: []const u8) !std.array_list.Managed(u32) {
        var result = std.array_list.Managed(u32).init(self.allocator);
        errdefer result.deinit();
        if (self.vocab.get(token)) |id| {
            try result.append(id);
            return result;
        }
        if (token.len == 0) return result;
        var boundaries = std.array_list.Managed(usize).init(self.allocator);
        defer boundaries.deinit();
        try boundaries.append(0);
        for (1..token.len + 1) |i| {
            try boundaries.append(i);
        }
        var did_merge = true;
        while (did_merge and boundaries.items.len > 2) {
            did_merge = false;
            var best_idx: ?usize = null;
            var best_rank: u32 = std.math.maxInt(u32);
            for (0..boundaries.items.len - 2) |i| {
                const start = boundaries.items[i];
                const end = boundaries.items[i + 2];
                const pair = token[start..end];
                if (self.vocab.get(pair)) |rank| {
                    if (rank < best_rank) {
                        best_rank = rank;
                        best_idx = i;
                    }
                }
            }
            if (best_idx) |i| {
                did_merge = true;
                _ = boundaries.orderedRemove(i + 1);
            }
        }
        const num_tokens = boundaries.items.len - 1;
        for (0..num_tokens) |i| {
            const start = boundaries.items[i];
            const end = boundaries.items[i + 1];
            const segment = token[start..end];
            if (self.vocab.get(segment)) |id| {
                try result.append(id);
            } else {
                std.debug.print("Warning: No ID found for segment: '{s}'\n", .{segment});
                try result.append(0);
            }
        }
        return result;
    }

    pub fn encode(self: *Self, text: []const u8) ![]const u32 {
        std.debug.print("\nInput to encode: {s}\n", .{text});
        var result = std.array_list.Managed(u32).init(self.allocator);
        errdefer result.deinit();
        var parts = try self.split_with_specials(text);
        defer {
            for (parts.items) |part| {
                self.allocator.free(part);
            }
            parts.deinit();
        }
        for (parts.items) |part| {
            if (self.vocab.get(part)) |id| {
                try result.append(id);
                continue;
            }
            var tokens = try self.split_with_pattern(part);
            defer {
                for (tokens.items) |token| {
                    self.allocator.free(token);
                }
                tokens.deinit();
            }
            for (tokens.items) |token| {
                var token_ids = try self.bpe_merges(token);
                defer token_ids.deinit();
                try result.appendSlice(token_ids.items);
            }
        }
        return result.toOwnedSlice();
    }

    pub fn decode(self: *Self, token_ids: []const u32) ![]const u8 {
        var result = std.array_list.Managed(u8).init(self.allocator);
        errdefer result.deinit();
        for (token_ids) |id| {
            if (self.id_to_token.get(id)) |token| {
                try result.appendSlice(token);
            } else {
                std.debug.print("Warning: Unknown token ID: {d}\n", .{id});
            }
        }
        return result.toOwnedSlice();
    }

    pub fn encode_chat(
        self: *Self,
        chat_format: ?[]const u8,
        replacements: []const []const u8,
    ) ![]const u32 {
        std.debug.print("\nReplacements #: {d}\nChat format: ", .{replacements.len});
        if (chat_format == null) {
            std.debug.print("null\n", .{});
            return self.encode(replacements[0]);
        }
        std.debug.print("{s}\n", .{chat_format.?});
        const format = chat_format.?;
        var replacement_parts = std.array_list.Managed(TextPart).init(self.allocator);
        defer replacement_parts.deinit();
        for (replacements) |replacement| {
            try replacement_parts.append(.{ .value = replacement });
        }
        const formatted = try format_dynamic(self.allocator, format, replacement_parts.items);
        defer self.allocator.free(formatted);
        return self.encode(formatted);
    }
};

test "Tokenizer round-trip" {
    std.debug.print("\n=== TOKENIZER.ZIG ===\n\n", .{});
    const allocator = std.testing.allocator;
    const pattern_part_1 = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|";
    const pattern_part_2 = "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|";
    const pattern_part_3 = "\\p{N}{1,3}|";
    const pattern_part_4 = " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|";
    const pattern_part_5 = "\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    const pattern_prefix = pattern_part_1 ++ pattern_part_2 ++ pattern_part_3;
    const pattern_suffix = pattern_part_4 ++ pattern_part_5;
    const pattern = pattern_prefix ++ pattern_suffix;
    const specials = [_][]const u8{
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>",
        "<|finetune_right_pad_id|>",
        "<|step_id|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eom_id|>",
        "<|eot_id|>",
        "<|python_tag|>",
    };

    const tokenizer_model_path = try model_path_for_test(allocator);
    defer allocator.free(tokenizer_model_path);
    var tokenizer = try Tokenizer.init_from_tik_token(
        allocator,
        pattern,
        tokenizer_model_path,
        &specials,
    );
    defer tokenizer.deinit();
    const text =
        \\<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        \\
        \\Cutting Knowledge Date: December 2023
        \\Today Date: 26 Jul 2024
        \\
        \\s<|eot_id|><|start_header_id|>user<|end_header_id|>
        \\
        \\s<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        \\
        \\
    ;
    std.debug.print("Original text: \"{s}\"\n\n", .{text});
    std.debug.print("1. Encoding text to token IDs...\n", .{});
    const token_ids = try tokenizer.encode(text);
    defer allocator.free(token_ids);
    std.debug.print("   Result: {d} tokens\n", .{token_ids});
    for (token_ids, 0..) |id, i| {
        if (tokenizer.id_to_token.get(id)) |token| {
            std.debug.print("   Token {d}: ID {d} = '{s}'\n", .{ i, id, token });
        } else {
            std.debug.print("   Token {d}: ID {d} = UNKNOWN\n", .{ i, id });
        }
    }
    std.debug.print("\n2. Decoding token IDs back to text...\n", .{});
    const decoded_text = try tokenizer.decode(token_ids);
    defer allocator.free(decoded_text);
    std.debug.print("   Result: \"{s}\"\n", .{decoded_text});
    std.debug.print("\n3. Verifying round-trip accuracy...\n", .{});
    const match = std.mem.eql(u8, text, decoded_text);
    std.debug.print("   Original and decoded text match: {}\n", .{match});
    try std.testing.expect(match);
    if (!match) {
        var i: usize = 0;
        const min_len = @min(text.len, decoded_text.len);
        while (i < min_len and text[i] == decoded_text[i]) {
            i += 1;
        }
        if (i < min_len) {
            const start = if (i > 10) i - 10 else 0;
            const end = @min(i + 10, min_len);
            std.debug.print("\nFirst difference at position {d}:\n", .{i});
            std.debug.print(
                "  Original: ...{s}[{c}]{s}...\n",
                .{ text[start..i], text[i], text[i + 1 .. end] },
            );
            std.debug.print(
                "  Decoded:  ...{s}[{c}]{s}...\n",
                .{ decoded_text[start..i], decoded_text[i], decoded_text[i + 1 .. end] },
            );
        } else if (text.len != decoded_text.len) {
            if (text.len > decoded_text.len) {
                std.debug.print("\nDecoded text is missing: '{s}'\n", .{text[min_len..]});
            } else {
                std.debug.print("\nDecoded text has extra: '{s}'\n", .{decoded_text[min_len..]});
            }
        }
    }
}
