const std = @import("std");

pub fn dequant_tensor(allocator: std.mem.Allocator, tensor_reader: *std.Io.Reader) !void {
    const header_size: usize = @intCast(try tensor_reader.takeInt(u64, .little));
    const header_data = try allocator.alloc(u8, header_size);
    defer allocator.free(header_data);

    try tensor_reader.readSliceAll(header_data);

    std.debug.print("Taille header {}\nContent: {s}\n", .{ header_size, header_data });
}
