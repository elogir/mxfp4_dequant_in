const std = @import("std");

pub fn dequant_tensor(tensor_reader: *std.Io.Reader) !void {
    std.debug.print("Slt\n", .{});
    _ = tensor_reader;
}
