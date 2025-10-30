const std = @import("std");
const json = std.json;

const TensorInfo = struct {
    name: []const u8,
    dtype: []const u8,
    shape: [4]u32,
    shape_len: usize,
    data_offsets: [2]u64,
    // THe following are not in the json
    is_mxfp4_blocks: bool = false,
    is_mxfp4_scales: bool = false,
    base_name: []const u8,
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
        else => {}, // ca devrait free ayuto
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

        var shape: [4]u32 = undefined;
        var shape_len: usize = 0;
        for (shape_array.items) |dim| {
            shape[shape_len] = @as(u32, @intCast(dim.integer));
            shape_len += 1;
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
            .shape_len = shape_len,
            .data_offsets = data_offsets,
            .is_mxfp4_blocks = is_blocks,
            .is_mxfp4_scales = is_scales,
            .base_name = base_name,
        });
    }

    return tensors.toOwnedSlice(allocator);
}

fn add_entry(allocator: std.mem.Allocator, shape: [4]u32, shape_len: usize, dtype: []const u8, data_offsets: [2]u64) !json.Value {
    var shape_arr: json.Array = .init(allocator);
    for (shape[0..shape_len]) |dim| {
        try shape_arr.append(json.Value{ .integer = @as(i64, @intCast(dim)) });
    }

    var offsets_arr: json.Array = .init(allocator);
    for (data_offsets) |size| {
        try offsets_arr.append(json.Value{ .integer = @as(i64, @intCast(size)) });
    }

    var obj: json.ObjectMap = .init(allocator);
    try obj.put("dtype", json.Value{ .string = dtype });
    try obj.put("shape", json.Value{ .array = shape_arr });
    try obj.put("data_offsets", json.Value{ .array = offsets_arr });

    return json.Value{ .object = obj };
}

fn dequant_header(blocks: *TensorInfo, new_start_offset: u64) TensorInfo {
    const d0: u32 = blocks.shape[0];
    const d1: u32 = blocks.shape[1];
    const d2: u32 = blocks.shape[2];
    const d3: u32 = blocks.shape[3]; // block size

    const dequant_size: u64 = @as(u64, d0) * @as(u64, d1) * @as(u64, d2) * @as(u64, d3 * 2) * 2;
    const new_end = new_start_offset + dequant_size;

    var new_shape: [4]u32 = undefined;
    new_shape[0] = d0;
    new_shape[1] = d2 * d3 * 2;
    new_shape[2] = d1;

    const new_data_offsets: [2]u64 = .{ new_start_offset, new_end };

    return TensorInfo{
        .name = blocks.base_name,
        .base_name = blocks.base_name,
        .dtype = "BF16",
        .shape = new_shape,
        .shape_len = 3,
        .data_offsets = new_data_offsets,
    };
}

fn transform_header(allocator: std.mem.Allocator, tensors: *const []TensorInfo, new_header: *json.ObjectMap) !void {
    var cumulative_delta: i64 = 0;
    var mxfp_pairs: std.StringHashMap(MxfpPair) = .init(allocator);
    defer mxfp_pairs.deinit();

    for (tensors.*) |*tensor| {
        const new_start_offset = @as(u64, @intCast(@as(i64, @intCast(tensor.data_offsets[0])) + cumulative_delta));

        const tensor_size = tensor.data_offsets[1] - tensor.data_offsets[0];

        if (tensor.is_mxfp4_blocks or tensor.is_mxfp4_scales) {
            // We find a mxfp4 tensor
            const item = try mxfp_pairs.getOrPut(tensor.base_name);
            if (!item.found_existing) {
                item.value_ptr.* = MxfpPair{};
            }

            if (tensor.is_mxfp4_blocks) {
                item.value_ptr.blocks = tensor;
            } else {
                item.value_ptr.scales = tensor;
            }

            cumulative_delta -= @as(i64, @intCast(tensor_size));

            if (item.value_ptr.isComplete()) {
                std.debug.print("Found a pair of mxfp4: {s}\n", .{tensor.base_name});

                const blocks = item.value_ptr.blocks.?;
                const dequanted_tensor = dequant_header(blocks, new_start_offset);

                const dequant_size = dequanted_tensor.data_offsets[1] - dequanted_tensor.data_offsets[0];
                cumulative_delta += @as(i64, @intCast(dequant_size));

                const new_entry = try add_entry(allocator, dequanted_tensor.shape, dequanted_tensor.shape_len, dequanted_tensor.dtype, dequanted_tensor.data_offsets);
                try new_header.put(tensor.base_name, new_entry);

                _ = mxfp_pairs.remove(tensor.base_name);
            }
        } else {
            // We find a regular tensor
            std.debug.print("Found a regular tensor: {s}\n", .{tensor.base_name});
            const new_end_offset = new_start_offset + tensor_size;
            std.debug.print("New offsets for tensor: [{}, {}] => [{}, {}]\n", .{ tensor.data_offsets[0], tensor.data_offsets[1], new_start_offset, new_end_offset });

            const new_offsets: [2]u64 = .{ new_start_offset, new_end_offset };

            const new_entry = try add_entry(allocator, tensor.shape, tensor.shape_len, tensor.dtype, new_offsets);
            try new_header.put(tensor.base_name, new_entry);
        }
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
            allocator.free(tensor.base_name);
        }
        allocator.free(tensors);
    }

    // Sort by offset
    std.mem.sort(TensorInfo, tensors, {}, sortTensorFn);

    // Now we have a sorted slice of tensor infos that we can process in a second pass

    var new_header: json.ObjectMap = .init(allocator);
    defer {
        var header_iter = new_header.iterator();
        while (header_iter.next()) |entry| {
            deinitJsonValue(allocator, entry.value_ptr);
        }
        new_header.deinit();
    }

    try transform_header(allocator, &tensors, &new_header);

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
