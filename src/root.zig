const std = @import("std");
const json = std.json;
const safetensors = @import("safetensors.zig");

// TEMP FUNC
fn writeHeaderToFile(allocator: std.mem.Allocator, header: *json.ObjectMap, path: []const u8) !void {
    const json_bytes = try json.Stringify.valueAlloc(allocator, json.Value{ .object = header.* }, .{ .whitespace = .minified });
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
// TEMP FUNC

pub fn dequant_safetensors(allocator: std.mem.Allocator, tensor_reader: *std.Io.Reader) !void {
    _ = allocator;

    var header_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer header_arena.deinit();
    const arena = header_arena.allocator();

    const new_header = try safetensors.parseHeader(arena, tensor_reader);
    try writeHeaderToFile(arena, new_header, "NEW_HEADER.json");
}
