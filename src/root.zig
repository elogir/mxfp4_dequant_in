const std = @import("std");
const json = std.json;

const BLOCK_SUFFIX = "_blocks";
const SCALE_SUFFIX = "_scales";

const TensorInfo = struct {
    name: []const u8,
    dtype: []const u8,
    shape: [4]u32,
    shape_len: usize,
    data_offsets: [2]u64,
    is_mxfp4_blocks: bool = false,
    is_mxfp4_scales: bool = false,
    base_name: []const u8,

    fn init(allocator: std.mem.Allocator, name: []const u8, tensor_obj: json.ObjectMap) !TensorInfo {
        const dtype = tensor_obj.get("dtype").?.string;
        const shape_result = parseShape(tensor_obj.get("shape").?.array);
        const data_offsets = parseDataOffsets(tensor_obj.get("data_offsets").?.array);

        const is_blocks = isMxfp4Blocks(name);
        const is_scales = isMxfp4Scales(name);
        const base_name = try getBaseName(allocator, name);

        return TensorInfo{
            .name = name,
            .dtype = dtype,
            .shape = shape_result.shape,
            .shape_len = shape_result.len,
            .data_offsets = data_offsets,
            .is_mxfp4_blocks = is_blocks,
            .is_mxfp4_scales = is_scales,
            .base_name = base_name,
        };
    }

    fn size(self: TensorInfo) u64 {
        return self.data_offsets[1] - self.data_offsets[0];
    }
};

const MxfpPair = struct {
    blocks: ?*TensorInfo = null,
    scales: ?*TensorInfo = null,

    fn isComplete(self: *const MxfpPair) bool {
        return self.blocks != null and self.scales != null;
    }
};

fn calculateDequantSize(blocks: *TensorInfo) u64 {
    const d0: u32 = blocks.shape[0];
    const d1: u32 = blocks.shape[1];
    const d2: u32 = blocks.shape[2];
    const d3: u32 = blocks.shape[3];

    return @as(u64, d0) * @as(u64, d1) * @as(u64, d2) * @as(u64, d3 * 2) * 2;
}

fn calculateDequantShape(blocks: *TensorInfo) struct { shape: [4]u32, len: usize } {
    var new_shape: [4]u32 = undefined;

    new_shape[0] = blocks.shape[0];
    new_shape[1] = blocks.shape[2] * blocks.shape[3] * 2;
    new_shape[2] = blocks.shape[1];

    return .{ .shape = new_shape, .len = 3 };
}

const DequantedTensor = struct {
    name: []const u8,
    dtype: []const u8,
    shape: [4]u32,
    shape_len: usize,
    data_offsets: [2]u64,

    fn init(blocks: *TensorInfo, start_offset: u64) DequantedTensor {
        const dequant_size = calculateDequantSize(blocks);
        const shape_result = calculateDequantShape(blocks);

        return DequantedTensor{
            .name = blocks.base_name,
            .dtype = "BF16",
            .shape = shape_result.shape,
            .shape_len = shape_result.len,
            .data_offsets = .{ start_offset, start_offset + dequant_size },
        };
    }
};

fn isMxfp4Blocks(name: []const u8) bool {
    return std.mem.endsWith(u8, name, BLOCK_SUFFIX);
}

fn isMxfp4Scales(name: []const u8) bool {
    return std.mem.endsWith(u8, name, SCALE_SUFFIX);
}

fn getBaseName(allocator: std.mem.Allocator, name: []const u8) ![]const u8 {
    if (isMxfp4Blocks(name)) {
        return try allocator.dupe(u8, name[0 .. name.len - "_blocks".len]);
    } else if (isMxfp4Scales(name)) {
        return try allocator.dupe(u8, name[0 .. name.len - "_scales".len]);
    }

    return try allocator.dupe(u8, name);
}

fn parseShape(shape_array: json.Array) struct { shape: [4]u32, len: usize } {
    var shape: [4]u32 = undefined;
    var len: usize = 0;

    for (shape_array.items) |dim| {
        shape[len] = @as(u32, @intCast(dim.integer));
        len += 1;
    }

    return .{ .shape = shape, .len = len };
}

fn parseDataOffsets(offsets_array: json.Array) [2]u64 {
    return [2]u64{
        @intCast(offsets_array.items[0].integer),
        @intCast(offsets_array.items[1].integer),
    };
}

fn loadTensorsFromHeader(allocator: std.mem.Allocator, parsed_header: *const json.Value) ![]TensorInfo {
    var tensors: std.ArrayList(TensorInfo) = .empty;

    const header = parsed_header.object;
    var iter = header.iterator();
    while (iter.next()) |entry| {
        const tensor_name = entry.key_ptr.*;
        if (std.mem.eql(u8, tensor_name, "__metadata__")) continue;

        const tensor_info = try TensorInfo.init(allocator, tensor_name, entry.value_ptr.object);
        try tensors.append(allocator, tensor_info);
    }

    return tensors.toOwnedSlice(allocator);
}

fn createJsonShapeArray(allocator: std.mem.Allocator, shape: [4]u32, shape_len: usize) !json.Array {
    var arr: json.Array = .init(allocator);

    for (shape[0..shape_len]) |dim| {
        try arr.append(json.Value{ .integer = @as(i64, @intCast(dim)) });
    }

    return arr;
}

fn createJsonOffsetsArray(allocator: std.mem.Allocator, data_offsets: [2]u64) !json.Array {
    var arr: json.Array = .init(allocator);

    for (data_offsets) |offset| {
        try arr.append(json.Value{ .integer = @as(i64, @intCast(offset)) });
    }

    return arr;
}

fn createTensorEntry(allocator: std.mem.Allocator, shape: [4]u32, shape_len: usize, dtype: []const u8, data_offsets: [2]u64) !json.Value {
    var obj: json.ObjectMap = .init(allocator);

    try obj.put("dtype", json.Value{ .string = dtype });
    try obj.put("shape", json.Value{ .array = try createJsonShapeArray(allocator, shape, shape_len) });
    try obj.put("data_offsets", json.Value{ .array = try createJsonOffsetsArray(allocator, data_offsets) });

    return json.Value{ .object = obj };
}

fn handleMxfp4Tensor(
    allocator: std.mem.Allocator,
    tensor: *TensorInfo,
    pairs: *std.StringHashMap(MxfpPair),
    offset: u64,
    new_header: *json.ObjectMap,
) !i64 {
    const tensor_size = tensor.size();
    var offset_delta: i64 = -@as(i64, @intCast(tensor_size));

    const item = try pairs.getOrPut(tensor.base_name);
    if (!item.found_existing) {
        item.value_ptr.* = MxfpPair{};
    }

    if (tensor.is_mxfp4_blocks) {
        item.value_ptr.blocks = tensor;
    } else {
        item.value_ptr.scales = tensor;
    }

    if (item.value_ptr.isComplete()) {
        std.debug.print("Found MXFP4 pair: {s}\n", .{tensor.base_name});

        const blocks = item.value_ptr.blocks.?;
        const dequanted = DequantedTensor.init(blocks, offset);
        const dequant_size = dequanted.data_offsets[1] - dequanted.data_offsets[0];

        offset_delta += @as(i64, @intCast(dequant_size));

        const entry = try createTensorEntry(allocator, dequanted.shape, dequanted.shape_len, dequanted.dtype, dequanted.data_offsets);
        try new_header.put(tensor.base_name, entry);

        _ = pairs.remove(tensor.base_name);
    }

    return offset_delta;
}

fn handleRegularTensor(
    allocator: std.mem.Allocator,
    tensor: *TensorInfo,
    offset: u64,
    new_header: *json.ObjectMap,
) !void {
    std.debug.print("Found regular tensor: {s}\n", .{tensor.base_name});

    const tensor_size = tensor.size();
    const new_offsets: [2]u64 = .{ offset, offset + tensor_size };

    const entry = try createTensorEntry(allocator, tensor.shape, tensor.shape_len, tensor.dtype, new_offsets);
    try new_header.put(tensor.base_name, entry);
}

/// La logique est la suivante:
/// On parse chaque tenseur un a un. Si celui ci est un tenseur mxfp4, on note la presence de la paire dans une hashmap.
/// Si les 2 tenseurs necessaires (scales + blocks) sont trouve, on ecrit alors le nouveau tenseur les nouvelles shapes
/// et les nouveaux offsets. Pour les tenseurs non-quantizes, on les ajoute normalement dans le nouveau header.
/// Nous gardons un tracking des differences de taille pour ne pas ecrire n'importe comment les offsets.
fn transformHeader(allocator: std.mem.Allocator, tensors: []TensorInfo, new_header: *json.ObjectMap) !void {
    var cumulative_delta: i64 = 0;
    var mxfp_pairs: std.StringHashMap(MxfpPair) = .init(allocator);
    defer mxfp_pairs.deinit();

    for (tensors) |*tensor| {
        // offset prenant en compte le decalage
        const current_offset = @as(u64, @intCast(@as(i64, @intCast(tensor.data_offsets[0])) + cumulative_delta));

        if (tensor.is_mxfp4_blocks or tensor.is_mxfp4_scales) {
            const delta = try handleMxfp4Tensor(allocator, tensor, &mxfp_pairs, current_offset, new_header);
            cumulative_delta += delta;
        } else {
            try handleRegularTensor(allocator, tensor, current_offset, new_header);
        }
    }
}

fn writeHeaderToFile(allocator: std.mem.Allocator, header: json.ObjectMap, path: []const u8) !void {
    const json_bytes = try json.Stringify.valueAlloc(allocator, json.Value{ .object = header }, .{ .whitespace = .minified });
    defer allocator.free(json_bytes);

    const header_size: u64 = json_bytes.len;

    const output_file = try std.fs.cwd().createFile(path, .{});
    defer output_file.close();

    var file_buffer: [0xFFFF]u8 = undefined;
    var file_writer = output_file.writer(&file_buffer);

    try file_writer.interface.writeInt(u64, header_size, .little);
    try file_writer.interface.writeAll(json_bytes);
    try file_writer.interface.flush();

    std.debug.print("New header written: {} bytes\n", .{header_size});
}

fn readHeaderData(allocator: std.mem.Allocator, reader: *std.Io.Reader) ![]u8 {
    const header_size: usize = @intCast(try reader.takeInt(u64, .little));
    const header_data = try allocator.alloc(u8, header_size);

    try reader.readSliceAll(header_data);

    return header_data;
}

fn sortTensorFn(_: void, lhs: TensorInfo, rhs: TensorInfo) bool {
    return lhs.data_offsets[0] < rhs.data_offsets[0];
}

fn parseHeader(tensor_reader: *std.Io.Reader) !void {
    var header_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator); // TODO: do not create it here
    defer header_arena.deinit();
    const arena = header_arena.allocator();

    const header_data = try readHeaderData(arena, tensor_reader);
    const parsed = try json.parseFromSliceLeaky(json.Value, arena, header_data, .{});

    const tensors = try loadTensorsFromHeader(arena, &parsed);
    std.mem.sort(TensorInfo, tensors, {}, sortTensorFn);

    var new_header: json.ObjectMap = .init(arena);

    if (parsed.object.get("__metadata__")) |metadata| {
        try new_header.put("__metadata__", metadata);
    }

    try transformHeader(arena, tensors, &new_header);

    try writeHeaderToFile(arena, new_header, "NEW_HEADER.json"); // TEMP CALL
}

pub fn dequant_safetensors(allocator: std.mem.Allocator, tensor_reader: *std.Io.Reader) !void {
    _ = allocator;
    try parseHeader(tensor_reader);
}
