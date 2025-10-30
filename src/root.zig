const std = @import("std");
const json = std.json;

const TensorInfo = struct {
    name: []const u8,
    dtype: []const u8,
    shape: std.ArrayList(i64),
    data_offsets: [2]usize,
    // THe following are not in the json
    is_mxfp4_blocks: bool,
    is_mxfp4_scales: bool,
    base_name: []const u8,

    fn deinit(self: *TensorInfo, allocator: std.mem.Allocator) void {
        self.shape.deinit(allocator);
    }
};

const MxfpPair = struct {
    blocks: ?*TensorInfo = null,
    scales: ?*TensorInfo = null,

    fn isComplete(self: *const MxfpPair) bool {
        return self.blocks != null and self.scales != null;
    }
};

fn isMxfp4Blocks(name: []const u8) bool {
    return std.mem.endsWith(u8, name, "_blocks");
}

fn isMxfp4Scales(name: []const u8) bool {
    return std.mem.endsWith(u8, name, "_scales");
}

fn getBaseName(allocator: std.mem.Allocator, name: []const u8) ![]const u8 {
    if (isMxfp4Blocks(name)) {
        return try allocator.dupe(u8, name[0 .. name.len - "_blocks".len]);
    } else if (isMxfp4Scales(name)) {
        return try allocator.dupe(u8, name[0 .. name.len - "_scales".len]);
    }
    return try allocator.dupe(u8, name);
}

fn sortTensorFn(_: void, lhs: TensorInfo, rhs: TensorInfo) bool {
    return lhs.data_offsets[0] < rhs.data_offsets[0];
}

fn deinitJsonValue(allocator: std.mem.Allocator, value: *json.Value) void {
    switch (value.*) {
        .object => |*obj| {
            var iter = obj.iterator();
            while (iter.next()) |entry| {
                deinitJsonValue(allocator, entry.value_ptr);
            }
            obj.deinit();
        },
        .array => |*arr| {
            for (arr.items) |*item| {
                deinitJsonValue(allocator, item);
            }
            arr.deinit();
        },
        else => unreachable, // dans notre cas en tout cas
    }
}

fn load_header(allocator: std.mem.Allocator, parsed_header: *const json.Parsed(json.Value)) ![]TensorInfo {
    var tensors: std.ArrayList(TensorInfo) = .empty;

    const header = parsed_header.value.object;
    var iter = header.iterator();
    while (iter.next()) |entry| {
        const tensor_name = entry.key_ptr.*;
        const tensor_info = entry.value_ptr.*;

        if (std.mem.eql(u8, tensor_name, "__metadata__")) continue;

        const dtype = tensor_info.object.get("dtype").?.string;
        const shape_array = tensor_info.object.get("shape").?.array;
        const offsets_array = tensor_info.object.get("data_offsets").?.array;

        var shape: std.ArrayList(i64) = .empty;
        for (shape_array.items) |dim| {
            try shape.append(allocator, dim.integer);
        }

        const data_offsets = [2]usize{
            @intCast(offsets_array.items[0].integer),
            @intCast(offsets_array.items[1].integer),
        };

        const is_blocks = isMxfp4Blocks(tensor_name);
        const is_scales = isMxfp4Scales(tensor_name);
        const base_name = try getBaseName(allocator, tensor_name);

        try tensors.append(allocator, .{
            .name = tensor_name,
            .dtype = dtype,
            .shape = shape,
            .data_offsets = data_offsets,
            .is_mxfp4_blocks = is_blocks,
            .is_mxfp4_scales = is_scales,
            .base_name = base_name,
        });
    }

    return tensors.toOwnedSlice(allocator);
}

fn transform_header(allocator: std.mem.Allocator, tensors: *const []TensorInfo) !void {
    var cumulative_delta: i64 = 0;
    var mxfp_pairs: std.StringHashMap(MxfpPair) = .init(allocator);
    defer mxfp_pairs.deinit();

    for (tensors.*) |*tensor| {
        const new_start_offset = @as(u64, @intCast(@as(i64, @intCast(tensor.data_offsets[0])) + cumulative_delta));
        _ = new_start_offset;

        if (tensor.is_mxfp4_blocks or tensor.is_mxfp4_scales) {
            const item = try mxfp_pairs.getOrPut(tensor.base_name);
            if (!item.found_existing) {
                item.value_ptr.* = MxfpPair{};
            }

            if (tensor.is_mxfp4_blocks) {
                item.value_ptr.blocks = tensor;
            } else {
                item.value_ptr.scales = tensor;
            }

            if (item.value_ptr.blocks != null and item.value_ptr.scales != null) {
                std.debug.print("Found a pair of mxfp4 {s}\n", .{tensor.base_name});
                _ = mxfp_pairs.remove(tensor.base_name);
            }
        }

        // Pour l'instant on a pas les 2 cas mais globalement pour un cas de base:
        cumulative_delta += @as(i64, @intCast(tensor.data_offsets[1])) - @as(i64, @intCast(tensor.data_offsets[0]));
    }
}

fn parse_header(allocator: std.mem.Allocator, tensor_reader: *std.Io.Reader) !void {
    const header_size: usize = @intCast(try tensor_reader.takeInt(u64, .little));
    const header_data = try allocator.alloc(u8, header_size);
    defer allocator.free(header_data);

    try tensor_reader.readSliceAll(header_data);

    const parsed: std.json.Parsed(json.Value) = try json.parseFromSlice(json.Value, allocator, header_data, .{});
    defer parsed.deinit();

    const tensors: []TensorInfo = try load_header(allocator, &parsed);
    defer { // free one by one
        for (tensors) |*tensor| {
            tensor.deinit(allocator);
            allocator.free(tensor.base_name);
        }
        allocator.free(tensors);
    }

    // Sort by offset
    std.mem.sort(TensorInfo, tensors, {}, sortTensorFn);

    // Now we have a sorted slice of tensor infos that we can process in a second pass

    var new_header = json.ObjectMap.init(allocator);
    defer {
        var header_iter = new_header.iterator();
        while (header_iter.next()) |entry| {
            deinitJsonValue(allocator, entry.value_ptr);
        }
        new_header.deinit();
    }

    try transform_header(allocator, &tensors);

    // Write new header to file
    var output_buffer: [0xFFFF]u8 = undefined;
    const output_file = try std.fs.cwd().createFile("NEW_HEADER.json", .{});
    defer output_file.close();
    var output_writer = output_file.writer(&output_buffer);

    // Remettre .minify / default apres tests
    try json.Stringify.value(json.Value{ .object = new_header }, .{ .whitespace = .indent_2 }, &output_writer.interface);
    try output_writer.interface.flush();
    std.debug.print("\n=== New header written to NEW_HEADER.json ===\n", .{});
}

pub fn dequant_tensor(allocator: std.mem.Allocator, tensor_reader: *std.Io.Reader) !void {
    try parse_header(allocator, tensor_reader);
}
