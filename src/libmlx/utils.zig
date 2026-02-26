//! utils.zig - Utility Functions
//!
//! Copyright 2025 Joe

const std = @import("std");

const MAX_FORMAT_SEGMENTS: usize = 128;

const IntegrityAlgo = enum {
    sha1,
    sha256,
};

const ExpectedDigest = struct {
    algo: IntegrityAlgo,
    hex: [64]u8,
    len: usize,

    fn slice(self: *const ExpectedDigest) []const u8 {
        return self.hex[0..self.len];
    }
};

pub const TextPart = struct {
    value: []const u8,
};

inline fn discard(_: anytype) void {}

fn default_io() std.Io {
    return std.Io.Threaded.global_single_threaded.io();
}

fn process_allocator() std.mem.Allocator {
    return std.heap.page_allocator;
}

fn is_hex_char(ch: u8) bool {
    return (ch >= '0' and ch <= '9') or
        (ch >= 'a' and ch <= 'f') or
        (ch >= 'A' and ch <= 'F');
}

fn digest_algo_for_hex_len(hex_len: u8) ?IntegrityAlgo {
    if (hex_len == 64) return .sha256;
    if (hex_len == 40) return .sha1;
    return null;
}

fn parse_digest_token(value: []const u8) ?ExpectedDigest {
    if (value.len == 0) {
        return null;
    }
    std.debug.assert(value.len > 0);

    var trimmed = std.mem.trim(u8, value, " \t\"");
    if (trimmed.len >= 2) {
        if (std.ascii.eqlIgnoreCase(trimmed[0..2], "W/")) {
            trimmed = std.mem.trim(u8, trimmed[2..], "\"");
        }
    }
    if (trimmed.len == 0) {
        return null;
    }

    const digest_len: u8 = @intCast(trimmed.len);
    const algo = digest_algo_for_hex_len(digest_len) orelse {
        return null;
    };
    std.debug.assert(trimmed.len == 40 or trimmed.len == 64);

    var parsed = ExpectedDigest{
        .algo = algo,
        .hex = undefined,
        .len = trimmed.len,
    };
    for (trimmed, 0..) |ch, idx| {
        if (!is_hex_char(ch)) {
            return null;
        }
        parsed.hex[idx] = std.ascii.toLower(ch);
    }
    return parsed;
}

fn header_name_matches(line: []const u8, header_name: []const u8) bool {
    if (line.len == 0) {
        return false;
    }
    if (header_name.len == 0) {
        return false;
    }
    std.debug.assert(line.len > 0);
    std.debug.assert(header_name.len > 0);

    if (line.len <= header_name.len) {
        return false;
    }
    std.debug.assert(line.len > header_name.len);
    if (line[header_name.len] != ':') {
        return false;
    }
    std.debug.assert(line[header_name.len] == ':');
    return std.ascii.eqlIgnoreCase(line[0..header_name.len], header_name);
}

fn digest_value_from_header_line(line: []const u8) ?[]const u8 {
    if (line.len == 0) {
        return null;
    }
    std.debug.assert(line.len > 0);

    if (header_name_matches(line, "x-linked-etag")) {
        return std.mem.trim(u8, line["x-linked-etag".len + 1 ..], " \t");
    }
    if (header_name_matches(line, "etag")) {
        return std.mem.trim(u8, line["etag".len + 1 ..], " \t");
    }
    if (header_name_matches(line, "x-xet-hash")) {
        return std.mem.trim(u8, line["x-xet-hash".len + 1 ..], " \t");
    }
    return null;
}

fn digest_from_header_line(raw_line: []const u8) ?ExpectedDigest {
    if (raw_line.len == 0) {
        return null;
    }
    std.debug.assert(raw_line.len > 0);
    const line = std.mem.trim(u8, raw_line, "\r");
    if (line.len == 0) {
        return null;
    }
    std.debug.assert(line.len > 0);
    const value = digest_value_from_header_line(line) orelse {
        return null;
    };
    if (value.len == 0) {
        return null;
    }
    std.debug.assert(value.len > 0);
    return parse_digest_token(value);
}

fn digest_from_header_name_value(name: []const u8, value: []const u8) ?ExpectedDigest {
    if (name.len == 0 or value.len == 0) {
        return null;
    }
    std.debug.assert(name.len > 0);
    std.debug.assert(value.len > 0);

    const is_digest_header = std.ascii.eqlIgnoreCase(name, "x-linked-etag") or
        std.ascii.eqlIgnoreCase(name, "etag") or
        std.ascii.eqlIgnoreCase(name, "x-xet-hash");
    if (!is_digest_header) {
        return null;
    }
    std.debug.assert(is_digest_header);

    const trimmed = std.mem.trim(u8, value, " \t");
    if (trimmed.len == 0) {
        return null;
    }
    std.debug.assert(trimmed.len > 0);
    return parse_digest_token(trimmed);
}

fn digest_header_priority(name: []const u8) ?u8 {
    if (name.len == 0) {
        return null;
    }
    std.debug.assert(name.len > 0);

    if (std.ascii.eqlIgnoreCase(name, "x-linked-etag")) {
        return 3;
    }
    if (std.ascii.eqlIgnoreCase(name, "etag")) {
        return 2;
    }
    if (std.ascii.eqlIgnoreCase(name, "x-xet-hash")) {
        return 1;
    }
    return null;
}

const DigestSelection = struct {
    best: ?ExpectedDigest = null,
    best_priority: u8 = 0,
};

fn update_digest_selection(
    selection: *DigestSelection,
    priority: u8,
    candidate: ExpectedDigest,
) void {
    if (selection.best == null) {
        selection.best = candidate;
        selection.best_priority = priority;
        return;
    }
    std.debug.assert(selection.best != null);

    if (priority > selection.best_priority) {
        selection.best = candidate;
        selection.best_priority = priority;
        return;
    }
    if (priority != selection.best_priority) {
        return;
    }
    if (selection.best) |current| {
        if (candidate.len > current.len) {
            selection.best = candidate;
        }
    } else {
        std.debug.assert(selection.best == null);
    }
}

fn parse_digest_from_response_head(head: std.http.Client.Response.Head) ?ExpectedDigest {
    const MAX_HEADER_LINES: usize = 512;
    var selection = DigestSelection{};
    var it = head.iterateHeaders();
    for (0..MAX_HEADER_LINES) |_| {
        const header = it.next() orelse break;
        const priority = digest_header_priority(header.name) orelse continue;
        const candidate = digest_from_header_name_value(header.name, header.value) orelse continue;
        update_digest_selection(&selection, priority, candidate);
    }

    if (selection.best) |digest| {
        std.debug.assert(digest.len == 40 or digest.len == 64);
    } else {
        std.debug.assert(selection.best == null);
    }
    return selection.best;
}

fn parse_digest_from_headers(headers: []const u8) ?ExpectedDigest {
    if (headers.len == 0) {
        return null;
    }
    std.debug.assert(headers.len > 0);

    const MAX_HEADER_LINES: usize = 512;
    var lines = std.mem.tokenizeScalar(u8, headers, '\n');
    var best: ?ExpectedDigest = null;

    for (0..MAX_HEADER_LINES) |_| {
        const raw_line = lines.next() orelse break;
        const candidate = digest_from_header_line(raw_line) orelse continue;
        if (best) |current| {
            if (candidate.len >= current.len) {
                best = candidate;
            }
        } else {
            best = candidate;
        }
    }

    if (best) |digest| {
        if (digest.len == 40) {
            return digest;
        }
        if (digest.len == 64) {
            return digest;
        }
        std.debug.assert(digest.len == 40 or digest.len == 64);
    }
    return null;
}

fn fetch_remote_head_digest(
    client: *std.http.Client,
    uri: std.Uri,
    url: []const u8,
) !?ExpectedDigest {
    if (@intFromPtr(client) == 0) {
        return error.RemoteDigestFetchFailed;
    }
    if (url.len == 0) {
        return error.RemoteDigestFetchFailed;
    }
    if (uri.host == null) {
        return error.RemoteDigestFetchFailed;
    }
    std.debug.assert(@intFromPtr(client) > 0);
    std.debug.assert(url.len > 0);
    std.debug.assert(uri.host != null);

    var request = client.request(.HEAD, uri, .{
        .redirect_behavior = .unhandled,
    }) catch {
        return error.RemoteDigestFetchFailed;
    };
    defer request.deinit();

    request.sendBodiless() catch {
        return error.RemoteDigestFetchFailed;
    };

    var redirect_buffer: [8 * 1024]u8 = undefined;
    const response = request.receiveHead(&redirect_buffer) catch {
        return error.RemoteDigestFetchFailed;
    };
    const status_class = response.head.status.class();
    if (status_class != .success and status_class != .redirect) {
        std.log.err(
            "Failed to fetch remote headers for '{s}': HTTP {d}",
            .{ url, @intFromEnum(response.head.status) },
        );
        return error.RemoteDigestFetchFailed;
    }
    std.debug.assert(status_class == .success or status_class == .redirect);

    return parse_digest_from_response_head(response.head);
}

fn fetch_remote_digest(allocator: std.mem.Allocator, url: []const u8) !?ExpectedDigest {
    if (url.len == 0) {
        return error.MissingRemoteDigest;
    }
    std.debug.assert(url.len > 0);

    const uri = std.Uri.parse(url) catch {
        return error.RemoteDigestFetchFailed;
    };
    const io = default_io();
    var client = std.http.Client{
        .allocator = allocator,
        .io = io,
    };
    defer client.deinit();
    return fetch_remote_head_digest(&client, uri, url);
}

fn hex_digit_lower(value: u8) u8 {
    std.debug.assert(value < 16);
    if (value < 10) {
        return '0' + value;
    }
    return 'a' + (value - 10);
}

fn bytes_to_hex_lower(out: []u8, bytes: []const u8) !void {
    if (out.len != bytes.len * 2) {
        return error.InvalidHexBuffer;
    }
    std.debug.assert(out.len == bytes.len * 2);
    for (bytes, 0..) |b, index| {
        out[index * 2] = hex_digit_lower((b >> 4) & 0xF);
        out[index * 2 + 1] = hex_digit_lower(b & 0xF);
    }
}

fn hash_read_limit(file_size: u64, chunk_size: u32) !u32 {
    if (chunk_size == 0) {
        return error.LocalDigestFailed;
    }
    std.debug.assert(chunk_size > 0);

    const chunk_size_u64: u64 = chunk_size;
    var chunks_needed_u64: u64 = 1;
    if (file_size == 0) {
        chunks_needed_u64 = 1;
    } else {
        chunks_needed_u64 = @divFloor(file_size - 1, chunk_size_u64) + 2;
    }
    if (chunks_needed_u64 > std.math.maxInt(u32)) {
        return error.LocalDigestFailed;
    }
    std.debug.assert(chunks_needed_u64 > 0);
    return @intCast(chunks_needed_u64);
}

const Sha1DigestPair = struct {
    raw_hex: [40]u8,
    git_blob_hex: [40]u8,
};

fn hash_file_sha1_variants_hex(
    file: std.Io.File,
    io: std.Io,
    out: *Sha1DigestPair,
) !void {
    const stat = file.stat(io) catch {
        return error.LocalDigestFailed;
    };
    if (stat.kind != .file) {
        return error.LocalDigestFailed;
    }
    const max_reads_u32 = try hash_read_limit(stat.size, 64 * 1024);
    const max_reads: usize = @intCast(max_reads_u32);
    std.debug.assert(stat.kind == .file);
    std.debug.assert(max_reads > 0);

    var prefix_buffer: [48]u8 = undefined;
    const prefix = std.fmt.bufPrint(&prefix_buffer, "blob {d}\x00", .{stat.size}) catch {
        return error.LocalDigestFailed;
    };

    var git_blob_hasher = std.crypto.hash.Sha1.init(.{});
    git_blob_hasher.update(prefix);

    var read_buffer: [64 * 1024]u8 = undefined;
    var offset: u64 = 0;
    var raw_hasher = std.crypto.hash.Sha1.init(.{});
    for (0..max_reads) |_| {
        const amt = file.readPositional(io, &.{read_buffer[0..]}, offset) catch {
            return error.LocalDigestFailed;
        };
        if (amt == 0) {
            break;
        }
        raw_hasher.update(read_buffer[0..amt]);
        git_blob_hasher.update(read_buffer[0..amt]);
        offset += @intCast(amt);
    }
    std.debug.assert(offset <= stat.size);

    var raw_digest_bytes: [std.crypto.hash.Sha1.digest_length]u8 = undefined;
    raw_hasher.final(&raw_digest_bytes);
    var git_blob_digest_bytes: [std.crypto.hash.Sha1.digest_length]u8 = undefined;
    git_blob_hasher.final(&git_blob_digest_bytes);

    out.* = .{
        .raw_hex = undefined,
        .git_blob_hex = undefined,
    };
    try bytes_to_hex_lower(&out.raw_hex, &raw_digest_bytes);
    try bytes_to_hex_lower(&out.git_blob_hex, &git_blob_digest_bytes);
}

fn hash_file_sha256_hex(file: std.Io.File, io: std.Io, out: *[64]u8) !void {
    const stat = file.stat(io) catch {
        return error.LocalDigestFailed;
    };
    if (stat.kind != .file) {
        return error.LocalDigestFailed;
    }
    const max_reads_u32 = try hash_read_limit(stat.size, 64 * 1024);
    const max_reads: usize = @intCast(max_reads_u32);
    std.debug.assert(stat.kind == .file);
    std.debug.assert(max_reads > 0);

    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    var read_buffer: [64 * 1024]u8 = undefined;
    var offset: u64 = 0;
    for (0..max_reads) |_| {
        const amt = file.readPositional(io, &.{read_buffer[0..]}, offset) catch {
            return error.LocalDigestFailed;
        };
        if (amt == 0) {
            break;
        }
        hasher.update(read_buffer[0..amt]);
        offset += @intCast(amt);
    }
    std.debug.assert(offset <= stat.size);

    var digest_bytes: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    hasher.final(&digest_bytes);
    try bytes_to_hex_lower(out, &digest_bytes);
}

fn verify_local_file_digest(
    allocator: std.mem.Allocator,
    local_path: []const u8,
    expected: ExpectedDigest,
) !void {
    _ = allocator;
    const expected_supported = expected.len == 40 or expected.len == 64;
    if (local_path.len == 0 or !expected_supported) {
        return error.LocalDigestFailed;
    }
    std.debug.assert(local_path.len > 0);
    std.debug.assert(expected_supported);

    const io = default_io();
    var file = std.Io.Dir.cwd().openFile(io, local_path, .{}) catch {
        return error.LocalDigestFailed;
    };
    defer file.close(io);

    var local_digest_40: [40]u8 = undefined;
    var local_digest_git_blob_40: [40]u8 = undefined;
    var local_digest_64: [64]u8 = undefined;
    var alternate_digest: ?[]const u8 = null;
    const local_digest = switch (expected.algo) {
        .sha1 => digest: {
            var digests: Sha1DigestPair = undefined;
            try hash_file_sha1_variants_hex(file, io, &digests);
            local_digest_40 = digests.raw_hex;
            local_digest_git_blob_40 = digests.git_blob_hex;
            alternate_digest = local_digest_git_blob_40[0..];
            break :digest local_digest_40[0..];
        },
        .sha256 => digest: {
            try hash_file_sha256_hex(file, io, &local_digest_64);
            break :digest local_digest_64[0..];
        },
    };

    const matches_local = std.ascii.eqlIgnoreCase(local_digest, expected.slice());
    var matches_alternate = false;
    if (alternate_digest) |digest| {
        matches_alternate = std.ascii.eqlIgnoreCase(digest, expected.slice());
    } else {
        std.debug.assert(alternate_digest == null);
    }
    if (!matches_local and !matches_alternate) {
        std.log.err(
            "Checksum mismatch for '{s}': expected {s}, got {s}",
            .{ local_path, expected.slice(), local_digest },
        );
        return error.IntegrityCheckFailed;
    }
}

fn verify_model_file_integrity(
    allocator: std.mem.Allocator,
    repo_name: []const u8,
    model_name: []const u8,
    filename: []const u8,
) !void {
    if (repo_name.len == 0) {
        return error.MissingRemoteDigest;
    }
    if (model_name.len == 0) {
        return error.MissingRemoteDigest;
    }
    if (filename.len == 0) {
        return error.MissingRemoteDigest;
    }
    std.debug.assert(repo_name.len > 0);
    std.debug.assert(model_name.len > 0);
    std.debug.assert(filename.len > 0);

    const local_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ model_name, filename });
    defer allocator.free(local_path);
    const url_path = try std.fmt.allocPrint(
        allocator,
        "https://huggingface.co/{s}/{s}/resolve/main/{s}",
        .{ repo_name, model_name, filename },
    );
    defer allocator.free(url_path);

    const expected = try fetch_remote_digest(allocator, url_path) orelse {
        std.log.err("Missing remote digest header for '{s}'", .{url_path});
        return error.MissingRemoteDigest;
    };
    try verify_local_file_digest(allocator, local_path, expected);
}

fn print_parsed_value(comptime T_: type, value: T_, allocator_: std.mem.Allocator) !void {
    const string = try std.json.Stringify.valueAlloc(
        allocator_,
        value,
        .{ .whitespace = .indent_2 },
    );
    defer allocator_.free(string);
    std.debug.print("\nParsed Value:\n", .{});
    std.debug.print("{s}\n", .{string});
}

fn print_field_differences(
    comptime T_: type,
    allocator_: std.mem.Allocator,
    json_text: []const u8,
) !void {
    _ = T_;
    if (json_text.len == 0) {
        return;
    }
    std.debug.assert(json_text.len > 0);

    var generic = try std.json.parseFromSlice(std.json.Value, allocator_, json_text, .{});
    defer generic.deinit();
    if (generic.value != .object) {
        return;
    }
    std.debug.assert(generic.value == .object);
    std.debug.assert(generic.value.object.count() >= 0);

    std.debug.print("\nJSON fields:\n", .{});
    var found_any = false;
    var iter = generic.value.object.iterator();
    while (iter.next()) |entry| {
        found_any = true;
        std.debug.print("  - {s}\n", .{entry.key_ptr.*});
    }
    if (found_any == false) {
        std.debug.print("  None\n", .{});
    }
}

pub fn download_model(
    allocator_: std.mem.Allocator,
    repo_name_: []const u8,
    model_name_: []const u8,
) !void {
    if (repo_name_.len == 0) {
        return error.InvalidModelName;
    }
    if (model_name_.len == 0) {
        return error.InvalidModelName;
    }
    std.debug.assert(repo_name_.len > 0);
    std.debug.assert(model_name_.len > 0);
    const custom_files = try get_model_files(allocator_, repo_name_, model_name_);
    defer if (custom_files) |files| {
        for (files) |file| allocator_.free(file.value);
        allocator_.free(files);
    };
    return download(allocator_, repo_name_, model_name_, custom_files);
}

fn get_model_files(
    allocator: std.mem.Allocator,
    repo_name: []const u8,
    model_name: []const u8,
) !?[]TextPart {
    if (repo_name.len == 0 or model_name.len == 0) {
        return null;
    }
    std.debug.assert(repo_name.len > 0);
    std.debug.assert(model_name.len > 0);

    const index_path = try std.fmt.allocPrint(
        allocator,
        "{s}/model.safetensors.index.json",
        .{model_name},
    );
    defer allocator.free(index_path);
    if (!file_exists(index_path)) {
        download(
            allocator,
            repo_name,
            model_name,
            &[_]TextPart{.{ .value = "model.safetensors.index.json" }},
        ) catch return null;
        if (!file_exists(index_path)) {
            return null;
        }
    }

    const io = default_io();
    const index_content = std.Io.Dir.cwd().readFileAlloc(
        io,
        index_path,
        allocator,
        .limited(10 * 1024 * 1024),
    ) catch return null;
    defer allocator.free(index_content);
    const pattern = try extract_pattern_from_index_json(allocator, index_content) orelse {
        return null;
    };
    defer allocator.free(pattern);

    var files = std.array_list.Managed(TextPart).init(allocator);
    errdefer {
        for (files.items) |file| allocator.free(file.value);
        files.deinit();
    }
    try files.append(.{ .value = try allocator.dupe(u8, "config.json") });
    try files.append(.{ .value = try allocator.dupe(u8, "tokenizer.json") });
    const ok = try append_weight_files_from_pattern(allocator, &files, pattern);
    if (!ok) {
        return null;
    }
    return try files.toOwnedSlice();
}

fn append_weight_files_from_pattern(
    allocator: std.mem.Allocator,
    files: *std.array_list.Managed(TextPart),
    pattern: []const u8,
) !bool {
    if (pattern.len == 0) {
        return false;
    }
    std.debug.assert(pattern.len > 0);

    if (std.mem.indexOf(u8, pattern, "-of-") != null) {
        return append_sharded_weight_files(allocator, files, pattern);
    }
    try files.append(.{ .value = try allocator.dupe(u8, pattern) });
    return true;
}

fn append_sharded_weight_files(
    allocator: std.mem.Allocator,
    files: *std.array_list.Managed(TextPart),
    pattern: []const u8,
) !bool {
    if (pattern.len == 0) {
        return false;
    }
    std.debug.assert(pattern.len > 0);

    const base_end = std.mem.indexOf(u8, pattern, "-00") orelse return false;
    const ext_start = std.mem.lastIndexOf(u8, pattern, ".") orelse return false;
    if (ext_start <= base_end) {
        return false;
    }
    std.debug.assert(ext_start > base_end);
    const count = parse_shard_count(pattern) orelse return false;
    if (count == 0) {
        return false;
    }
    std.debug.assert(count > 0);
    const shard_count: usize = @intCast(count);
    const base = pattern[0..base_end];
    const ext = pattern[ext_start..];
    for (1..shard_count + 1) |i| {
        const filename = try std.fmt.allocPrint(
            allocator,
            "{s}-{:0>5}-of-{:0>5}{s}",
            .{ base, i, shard_count, ext },
        );
        try files.append(.{ .value = filename });
    }
    return true;
}

fn parse_shard_count(pattern: []const u8) ?u32 {
    if (pattern.len == 0) {
        return null;
    }
    std.debug.assert(pattern.len > 0);

    const marker_idx = std.mem.indexOf(u8, pattern, "-of-") orelse return null;
    const total_part = pattern[marker_idx + 4 ..];
    if (total_part.len == 0) {
        return null;
    }
    std.debug.assert(total_part.len > 0);

    const dash_idx = std.mem.indexOf(u8, total_part, "-") orelse
        std.mem.indexOf(u8, total_part, ".") orelse return null;
    if (dash_idx == 0) {
        return null;
    }
    std.debug.assert(dash_idx > 0);

    const count_str = total_part[0..dash_idx];
    const count = std.fmt.parseInt(u32, count_str, 10) catch return null;
    if (count == 0) {
        return null;
    }
    std.debug.assert(count > 0);
    return count;
}

fn extract_pattern_from_index_json(
    allocator: std.mem.Allocator,
    index_content: []const u8,
) !?[]u8 {
    if (index_content.len == 0) {
        return null;
    }
    std.debug.assert(index_content.len > 0);

    var parsed = std.json.parseFromSlice(
        std.json.Value,
        allocator,
        index_content,
        .{},
    ) catch return null;
    defer parsed.deinit();

    const weight_map = parsed.value.object.get("weight_map") orelse return null;
    if (weight_map != .object) {
        return null;
    }
    if (weight_map.object.count() == 0) {
        return null;
    }
    var iter = weight_map.object.iterator();
    const first_entry = iter.next() orelse return null;
    if (first_entry.value_ptr.* != .string) {
        return null;
    }
    const pattern = first_entry.value_ptr.*.string;
    if (pattern.len == 0) {
        return null;
    }
    std.debug.assert(pattern.len > 0);
    return try allocator.dupe(u8, pattern);
}

fn verify_integrity_for_filenames(
    allocator: std.mem.Allocator,
    repo_name: []const u8,
    model_name: []const u8,
    filenames: []const TextPart,
) !void {
    if (repo_name.len == 0 or model_name.len == 0) {
        return error.InvalidModelName;
    }
    std.debug.assert(repo_name.len > 0);
    std.debug.assert(model_name.len > 0);
    for (filenames) |file_name| {
        try verify_model_file_integrity(
            allocator,
            repo_name,
            model_name,
            file_name.value,
        );
    }
}

pub fn download(
    allocator: std.mem.Allocator,
    repo_name: []const u8,
    model_name: []const u8,
    file_names: ?[]const TextPart,
) !void {
    if (repo_name.len == 0) {
        return error.InvalidModelName;
    }
    if (model_name.len == 0) {
        return error.InvalidModelName;
    }
    std.debug.assert(repo_name.len > 0);
    std.debug.assert(model_name.len > 0);

    const io = default_io();
    discard(std.Io.Dir.cwd().createDirPathStatus(io, model_name, .default_dir) catch |err| {
        std.log.err("Failed to create directory '{s}': {s}", .{ model_name, @errorName(err) });
        return err;
    });

    var args = std.array_list.Managed([]const u8).init(allocator);
    defer args.deinit();
    try args.append("curl");
    try args.append("--location");
    try args.append("--parallel");
    var paths_to_free = std.array_list.Managed([]const u8).init(allocator);
    defer {
        for (paths_to_free.items) |path| {
            allocator.free(path);
        }
        paths_to_free.deinit();
    }
    var all_exist = true;
    const default_filenames = [_]TextPart{
        .{ .value = "model.safetensors" },
        .{ .value = "config.json" },
        .{ .value = "tokenizer.json" },
    };
    const filenames: []const TextPart = file_names orelse &default_filenames;
    for (filenames) |file_name| {
        const filename = file_name.value;
        const local_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ model_name, filename });
        try paths_to_free.append(local_path);
        if (file_exists(local_path)) {
            std.debug.print("File '{s}' already exists. Skipping download.\n", .{local_path});
        } else {
            all_exist = false;
            const url_path = try std.fmt.allocPrint(
                allocator,
                "https://huggingface.co/{s}/{s}/resolve/main/{s}",
                .{ repo_name, model_name, filename },
            );
            try paths_to_free.append(url_path);
            try args.append(url_path);
            try args.append("-o");
            try args.append(local_path);
        }
    }

    if (all_exist) {
        std.debug.print("All files already exist. No download needed.\n", .{});
    } else {
        const proc_allocator = process_allocator();
        const result = try std.process.run(proc_allocator, io, .{
            .argv = args.items,
        });
        defer proc_allocator.free(result.stdout);
        defer proc_allocator.free(result.stderr);
        if (result.term != .exited) {
            return error.DownloadFailed;
        }
        if (result.term.exited != 0) {
            std.log.err(
                "Download failed with exit code: {d}\n{s}",
                .{ result.term.exited, result.stderr },
            );
            return error.DownloadFailed;
        }
    }

    try verify_integrity_for_filenames(allocator, repo_name, model_name, filenames);
}

pub fn format_dynamic(
    allocator: std.mem.Allocator,
    chat_fmt: []const u8,
    replacements: []const TextPart,
) ![]const u8 {
    std.debug.assert(chat_fmt.len > 0);
    std.debug.assert(replacements.len <= MAX_FORMAT_SEGMENTS);
    var segments: [MAX_FORMAT_SEGMENTS][]const u8 = undefined;
    var segment_count: usize = 0;
    var cursor: usize = 0;

    for (0..MAX_FORMAT_SEGMENTS) |_| {
        if (std.mem.indexOfPos(u8, chat_fmt, cursor, "{s}")) |idx| {
            segments[segment_count] = chat_fmt[cursor..idx];
            segment_count += 1;
            cursor = idx + 3;
            continue;
        }
        segments[segment_count] = chat_fmt[cursor..];
        segment_count += 1;
        cursor = chat_fmt.len;
        break;
    }

    if (cursor != chat_fmt.len) {
        return error.TemplateTooManySegments;
    }
    if (segment_count == 0) {
        return error.ReplacementCountMismatch;
    }

    const expected_replacements = segment_count - 1;
    if (replacements.len != expected_replacements) {
        return error.ReplacementCountMismatch;
    }

    var result = std.array_list.Managed(u8).init(allocator);
    errdefer result.deinit();

    for (0..expected_replacements) |idx| {
        const segment = segments[idx];
        const replacement = replacements[idx].value;
        try result.appendSlice(segment);
        try result.appendSlice(replacement);
    }
    try result.appendSlice(segments[expected_replacements]);
    return try result.toOwnedSlice();
}

pub fn load_audio(allocator: std.mem.Allocator, audio_file: []const u8) ![]f32 {
    if (audio_file.len == 0) {
        return error.AudioFileNotFound;
    }
    std.debug.assert(audio_file.len > 0);
    const io = default_io();
    std.Io.Dir.cwd().access(io, audio_file, .{}) catch return error.AudioFileNotFound;
    std.debug.print("\nAUDIO: {s}\n", .{audio_file});
    const args = [_][]const u8{
        "ffmpeg",
        "-nostdin",
        "-threads",
        "4",
        "-i",
        audio_file,
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-loglevel",
        "error",
        "-",
    };
    const proc_allocator = process_allocator();
    const process = try std.process.run(proc_allocator, io, .{
        .argv = &args,
        .stderr_limit = .limited(4096),
    });
    defer proc_allocator.free(process.stdout);
    defer proc_allocator.free(process.stderr);

    switch (process.term) {
        .exited => |code| {
            if (code != 0) {
                std.log.err("Failed to load audio: {s}", .{process.stderr});
                return error.FfmpegFailed;
            }
        },
        else => {
            return error.FfmpegFailed;
        },
    }
    if (process.stdout.len % 2 != 0) {
        return error.InvalidAudioData;
    }
    std.debug.assert(process.stdout.len % 2 == 0);
    std.debug.assert(process.stderr.len <= 4096);

    var float_samples = std.array_list.Managed(f32).init(allocator);
    defer float_samples.deinit();
    try float_samples.ensureTotalCapacity(process.stdout.len / 2);
    var i: usize = 0;
    while (i + 1 < process.stdout.len) : (i += 2) {
        const lo = @as(i16, process.stdout[i]);
        const hi = @as(i16, process.stdout[i + 1]);
        const sample = lo | (hi << 8);
        try float_samples.append(@as(f32, @floatFromInt(sample)) / 32768.0);
    }
    return float_samples.toOwnedSlice();
}

pub fn load_json(
    comptime T: type,
    allocator: std.mem.Allocator,
    filename: []const u8,
    verbose: bool,
) !std.json.Parsed(T) {
    if (filename.len == 0) {
        return error.FileNotFound;
    }
    std.debug.assert(filename.len > 0);
    const io = default_io();
    const json_string = try std.Io.Dir.cwd().readFileAlloc(
        io,
        filename,
        allocator,
        .limited(10 * 1024 * 1024),
    );
    defer allocator.free(json_string);
    const parsed = try std.json.parseFromSlice(T, allocator, json_string, .{
        .ignore_unknown_fields = true,
    });
    if (!verbose) {
        return parsed;
    }

    try print_field_differences(T, allocator, json_string);
    try print_parsed_value(T, parsed.value, allocator);
    return parsed;
}

pub fn load_config_json(
    comptime T: type,
    allocator: std.mem.Allocator,
    model_path: []const u8,
    verbose: bool,
) !std.json.Parsed(T) {
    if (model_path.len == 0) {
        return error.FileNotFound;
    }
    std.debug.assert(model_path.len > 0);
    var buf: [1024]u8 = undefined;
    const path_config = try std.fmt.bufPrintZ(&buf, "{s}/config.json", .{model_path});
    var parsed = try load_json(T, allocator, path_config, verbose);
    if (@hasField(T, "eos_token_id")) {
        if (@TypeOf(parsed.value.eos_token_id) == u32) {
            var eos_ids = try allocator.alloc(u32, 1);
            eos_ids[0] = parsed.value.eos_token_id;
            parsed.value.eos_token_ids = eos_ids;
        } else {
            parsed.value.eos_token_ids = try allocator.dupe(u32, parsed.value.eos_token_id);
        }
    }
    return parsed;
}

fn file_exists(path: []const u8) bool {
    if (path.len == 0) {
        return false;
    }
    std.debug.assert(path.len > 0);
    const io = default_io();
    std.Io.Dir.cwd().access(io, path, .{}) catch return false;
    return true;
}
