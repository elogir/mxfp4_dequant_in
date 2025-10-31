const std = @import("std");
const json = std.json;
const safetensors = @import("safetensors.zig");
const mxfp4 = @import("mxfp4.zig");
const dequant = @import("dequant.zig");

// TEMP FUNC
fn writeHeaderToFile(allocator: std.mem.Allocator, header: *json.ObjectMap, tensor_writer: *std.Io.Writer) !void {
    const json_bytes = try json.Stringify.valueAlloc(allocator, json.Value{ .object = header.* }, .{ .whitespace = .minified });
    defer allocator.free(json_bytes);
    const header_size: u64 = json_bytes.len;

    try tensor_writer.writeInt(u64, header_size, .little);
    try tensor_writer.writeAll(json_bytes);
    try tensor_writer.flush();

    std.debug.print("New header written: {} bytes\n", .{header_size});
}
// TEMP FUNC

pub fn dequant_safetensors(allocator: std.mem.Allocator, arena: std.mem.Allocator, tensor_reader: *std.Io.Reader, buffer: []u8) !dequant.Reader {
    // var header_arena = std.heap.ArenaAllocator.init(allocator);
    // const arena = header_arena.allocator();

    // const new_header = try safetensors.parseHeader(arena, tensor_reader);
    // try writeHeaderToFile(arena, new_header.new_json, tensor_writer);
    // try mxfp4.processTensors(allocator, new_header.old_header, tensor_reader, tensor_writer);

    // header_arena.deinit();

    return dequant.Reader.init(allocator, arena, tensor_reader, buffer);
}
