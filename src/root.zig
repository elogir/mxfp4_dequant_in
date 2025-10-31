const std = @import("std");
const json = std.json;
const safetensors = @import("safetensors.zig");

// TEMP FUNC
fn writeHeaderToFile(allocator: std.mem.Allocator, header: *json.ObjectMap, tensor_writer: *std.Io.Writer) !void {
    const json_bytes = try json.Stringify.valueAlloc(allocator, json.Value{ .object = header.* }, .{ .whitespace = .minified });
    const header_size: u64 = json_bytes.len;

    try tensor_writer.writeInt(u64, header_size, .little);
    try tensor_writer.writeAll(json_bytes);
    try tensor_writer.flush();

    std.debug.print("New header written: {} bytes\n", .{header_size});
}
// TEMP FUNC

pub fn dequant_safetensors(allocator: std.mem.Allocator, tensor_reader: *std.Io.Reader, tensor_writer: *std.Io.Writer) !void {
    _ = allocator;

    var header_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer header_arena.deinit();
    const arena = header_arena.allocator();

    const new_header = try safetensors.parseHeader(arena, tensor_reader);
    try writeHeaderToFile(arena, new_header.new_json, tensor_writer);
}
