const std = @import("std");
const json = std.json;

const TensorInfo = struct {
    name: []const u8,
    dtype: []const u8,
    shape: std.ArrayList(i64),
    data_offsets: [2]u64,
    is_mxfp4_blocks: bool,
    is_mxfp4_scales: bool,
    base_name: []const u8, // For MXFP4 tensors, the name without _blocks/_scales suffix

    fn deinit(self: *TensorInfo, allocator: std.mem.Allocator) void {
        self.shape.deinit(allocator);
    }
};

const MxfpPair = struct {
    blocks: ?TensorInfo = null,
    scales: ?TensorInfo = null,

    fn isComplete(self: *const MxfpPair) bool {
        return self.blocks != null and self.scales != null;
    }

    fn deinit(self: *MxfpPair, allocator: std.mem.Allocator) void {
        if (self.blocks) |*b| {
            var blocks_mut = b.*;
            blocks_mut.deinit(allocator);
            allocator.free(blocks_mut.base_name);
        }
        if (self.scales) |*s| {
            var scales_mut = s.*;
            scales_mut.deinit(allocator);
            allocator.free(scales_mut.base_name);
        }
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

fn calculateDequantSize(shape_blocks: []const i64) u64 {
    // MXFP4: blocks shape is [d0, d1, d2, block_size_bytes]
    // Each byte contains 2 FP4 values (4 bits each)
    // Dequantized output will be BF16 (2 bytes per value)
    // New shape: [d0, d2*(block_size_bytes*2), d1]

    if (shape_blocks.len != 4) {
        // Fallback for non-4D shapes
        var total_elements: u64 = 1;
        for (shape_blocks, 0..) |dim, i| {
            if (i == shape_blocks.len - 1) {
                // Last dimension: block_size_bytes → block_size_bytes * 2 FP4 values
                total_elements *= @intCast(dim * 2);
            } else {
                total_elements *= @intCast(dim);
            }
        }
        return total_elements * 2; // BF16 = 2 bytes per element
    }

    // 4D shape transformation: [d0, d1, d2, d3] -> [d0, d2*(d3*2), d1]
    const d0: u64 = @intCast(shape_blocks[0]);
    const d1: u64 = @intCast(shape_blocks[1]);
    const d2: u64 = @intCast(shape_blocks[2]);
    const d3: u64 = @intCast(shape_blocks[3]); // block size in bytes
    const block_values = d3 * 2; // 2 FP4 values per byte

    const total_elements = d0 * (d2 * block_values) * d1;

    // BF16 = 2 bytes per element
    return total_elements * 2;
}

fn lessThanOffset(context: void, a: TensorInfo, b: TensorInfo) bool {
    _ = context;
    return a.data_offsets[0] < b.data_offsets[0];
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
            arr.deinit(); // json.Array is Managed, deinit takes no args
        },
        else => {},
    }
}

fn parse_header(allocator: std.mem.Allocator, tensor_reader: *std.Io.Reader) !void {
    const header_size: usize = @intCast(try tensor_reader.takeInt(u64, .little));
    const header_data = try allocator.alloc(u8, header_size);
    defer allocator.free(header_data);

    try tensor_reader.readSliceAll(header_data);

    const parsed = try json.parseFromSlice(json.Value, allocator, header_data, .{});
    defer parsed.deinit();

    // First pass: collect all tensors
    const TensorList = std.ArrayList(TensorInfo);
    var tensors: TensorList = .{};
    defer {
        for (tensors.items) |*tensor| {
            tensor.deinit(allocator);
            allocator.free(tensor.base_name);
        }
        tensors.deinit(allocator);
    }

    const header = parsed.value.object;
    var iter = header.iterator();
    while (iter.next()) |entry| {
        const tensor_name = entry.key_ptr.*;
        const tensor_info = entry.value_ptr.*;

        if (std.mem.eql(u8, tensor_name, "__metadata__")) continue;

        const dtype = tensor_info.object.get("dtype").?.string;
        const shape_array = tensor_info.object.get("shape").?.array;
        const offsets_array = tensor_info.object.get("data_offsets").?.array;

        const ShapeList = std.ArrayList(i64);
        var shape: ShapeList = .{};
        for (shape_array.items) |dim| {
            try shape.append(allocator, dim.integer);
        }

        const data_offsets = [2]u64{
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

    // Sort by offset
    std.mem.sort(TensorInfo, tensors.items, {}, lessThanOffset);

    std.debug.print("\n=== Sorted tensors by offset ===\n", .{});
    for (tensors.items) |tensor| {
        std.debug.print("{s}: offset=[{}, {}], is_blocks={}, is_scales={}\n", .{
            tensor.name,
            tensor.data_offsets[0],
            tensor.data_offsets[1],
            tensor.is_mxfp4_blocks,
            tensor.is_mxfp4_scales,
        });
    }

    // Second pass: process tensors with HashMap approach
    var cumulative_delta: i64 = 0;
    var new_header = json.ObjectMap.init(allocator);
    defer {
        var header_iter = new_header.iterator();
        while (header_iter.next()) |entry| {
            deinitJsonValue(allocator, entry.value_ptr);
        }
        new_header.deinit();
    }

    const PairMap = std.StringHashMap(MxfpPair);
    var mxfp_pairs = PairMap.init(allocator);
    defer {
        var pair_iter = mxfp_pairs.iterator();
        while (pair_iter.next()) |entry| {
            var pair = entry.value_ptr.*;
            pair.deinit(allocator);
        }
        mxfp_pairs.deinit();
    }

    for (tensors.items) |tensor| {
        // Calculate new offset with cumulative delta
        const new_start_offset = @as(u64, @intCast(@as(i64, @intCast(tensor.data_offsets[0])) + cumulative_delta));

        if (tensor.is_mxfp4_blocks or tensor.is_mxfp4_scales) {
            // Get or create MXFP4 pair entry in HashMap
            const gop = try mxfp_pairs.getOrPut(tensor.base_name);
            if (!gop.found_existing) {
                gop.value_ptr.* = MxfpPair{};
            }

            // Store blocks or scales in the pair
            if (tensor.is_mxfp4_blocks) {
                gop.value_ptr.blocks = tensor;
            } else {
                gop.value_ptr.scales = tensor;
            }

            // Check if pair is complete
            if (gop.value_ptr.isComplete()) {
                const pair = gop.value_ptr.*;
                const blocks = pair.blocks.?;
                const scales = pair.scales.?;

                std.debug.print("\n=== Processing MXFP4 pair: {s} ===\n", .{tensor.base_name});
                std.debug.print("  blocks_shape: [", .{});
                for (blocks.shape.items, 0..) |dim, j| {
                    if (j > 0) std.debug.print(", ", .{});
                    std.debug.print("{}", .{dim});
                }
                std.debug.print("]\n  scales_shape: [", .{});
                for (scales.shape.items, 0..) |dim, j| {
                    if (j > 0) std.debug.print(", ", .{});
                    std.debug.print("{}", .{dim});
                }
                std.debug.print("]\n", .{});

                // Calculate sizes and delta
                const blocks_size = blocks.data_offsets[1] - blocks.data_offsets[0];
                const scales_size = scales.data_offsets[1] - scales.data_offsets[0];
                const original_size = blocks_size + scales_size;
                const dequant_size = calculateDequantSize(blocks.shape.items);
                const size_delta: i64 = @as(i64, @intCast(dequant_size)) - @as(i64, @intCast(original_size));

                std.debug.print("  Original size: blocks={} + scales={} = {}\n", .{ blocks_size, scales_size, original_size });
                std.debug.print("  Dequantized size: {}\n", .{dequant_size});
                std.debug.print("  Delta: {}\n", .{size_delta});

                // Use the earliest offset (blocks or scales)
                const pair_start_offset = @min(blocks.data_offsets[0], scales.data_offsets[0]);
                const final_start_offset = @as(u64, @intCast(@as(i64, @intCast(pair_start_offset)) + cumulative_delta));
                const final_end_offset = final_start_offset + dequant_size;

                std.debug.print("  New offsets: [{}, {}]\n", .{ final_start_offset, final_end_offset });

                // Build new shape: [d0, d1, d2, block_size_bytes] -> [d0, d2*(block_size_bytes*2), d1]
                var new_shape = json.Array.init(allocator);
                if (blocks.shape.items.len == 4) {
                    const d0 = blocks.shape.items[0];
                    const d1 = blocks.shape.items[1];
                    const d2 = blocks.shape.items[2];
                    const d3 = blocks.shape.items[3];
                    const block_values = d3 * 2; // 2 FP4 values per byte

                    try new_shape.append(json.Value{ .integer = d0 });
                    try new_shape.append(json.Value{ .integer = d2 * block_values });
                    try new_shape.append(json.Value{ .integer = d1 });

                    std.debug.print("  New shape: [{}, {}, {}] (block_size={} bytes → {} FP4 values)\n", .{ d0, d2 * block_values, d1, d3, block_values });
                } else {
                    for (blocks.shape.items, 0..) |dim, j| {
                        if (j == blocks.shape.items.len - 1) {
                            try new_shape.append(json.Value{ .integer = dim * 2 });
                        } else {
                            try new_shape.append(json.Value{ .integer = dim });
                        }
                    }
                }

                // Build tensor entry
                var tensor_obj = json.ObjectMap.init(allocator);
                try tensor_obj.put("dtype", json.Value{ .string = "BF16" });
                try tensor_obj.put("shape", json.Value{ .array = new_shape });

                var offsets_array = json.Array.init(allocator);
                try offsets_array.append(json.Value{ .integer = @intCast(final_start_offset) });
                try offsets_array.append(json.Value{ .integer = @intCast(final_end_offset) });
                try tensor_obj.put("data_offsets", json.Value{ .array = offsets_array });

                try new_header.put(tensor.base_name, json.Value{ .object = tensor_obj });

                // Update cumulative delta
                cumulative_delta += size_delta;

                // Clean up pair from HashMap
                _ = mxfp_pairs.remove(tensor.base_name);
            }
        } else {
            // Regular (non-MXFP4) tensor
            std.debug.print("\n=== Processing regular tensor: {s} ===\n", .{tensor.name});

            const original_size = tensor.data_offsets[1] - tensor.data_offsets[0];
            const new_end_offset = new_start_offset + original_size;

            std.debug.print("  Offsets: [{}, {}] -> [{}, {}]\n", .{
                tensor.data_offsets[0],
                tensor.data_offsets[1],
                new_start_offset,
                new_end_offset,
            });

            // Build shape array
            var shape_array = json.Array.init(allocator);
            for (tensor.shape.items) |dim| {
                try shape_array.append(json.Value{ .integer = dim });
            }

            // Build tensor entry
            var tensor_obj = json.ObjectMap.init(allocator);
            try tensor_obj.put("dtype", json.Value{ .string = tensor.dtype });
            try tensor_obj.put("shape", json.Value{ .array = shape_array });

            var offsets_array = json.Array.init(allocator);
            try offsets_array.append(json.Value{ .integer = @intCast(new_start_offset) });
            try offsets_array.append(json.Value{ .integer = @intCast(new_end_offset) });
            try tensor_obj.put("data_offsets", json.Value{ .array = offsets_array });

            try new_header.put(tensor.name, json.Value{ .object = tensor_obj });
        }
    }

    // Check for incomplete MXFP4 pairs
    if (mxfp_pairs.count() > 0) {
        std.debug.print("\nWARNING: Found incomplete MXFP4 pairs:\n", .{});
        var pair_iter = mxfp_pairs.iterator();
        while (pair_iter.next()) |entry| {
            std.debug.print("  - {s}: blocks={}, scales={}\n", .{
                entry.key_ptr.*,
                entry.value_ptr.blocks != null,
                entry.value_ptr.scales != null,
            });
        }
    }

    std.debug.print("\n=== Final cumulative delta: {} ===\n", .{cumulative_delta});

    // Write new header to file
    var output_buffer: [0xFFFF]u8 = undefined;
    const output_file = try std.fs.cwd().createFile("NEW_HEADER.json", .{});
    defer output_file.close();
    var output_writer = output_file.writer(&output_buffer);

    try json.Stringify.value(json.Value{ .object = new_header }, .{ .whitespace = .indent_2 }, &output_writer.interface);
    try output_writer.interface.flush();
    std.debug.print("\n=== New header written to NEW_HEADER.json ===\n", .{});
}

pub fn dequant_tensor(allocator: std.mem.Allocator, tensor_reader: *std.Io.Reader) !void {
    try parse_header(allocator, tensor_reader);
}
