const std = @import("std");
const mxfp4Loader = @import("mxfp4Loader");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len != 2) {
        std.debug.print("Usage: {s} <safetensors_file>", .{args[0]});
        return error.InvalidArguments;
    }

    const safetensors_path = args[1];
    const safetensors_file = try std.fs.cwd().openFile(safetensors_path, .{});
    defer safetensors_file.close();

    var safetensors_buf: [4096]u8 = undefined;
    var safetensors_reader = safetensors_file.reader(&safetensors_buf);

    try mxfp4Loader.dequant_safetensors(allocator, &safetensors_reader.interface);
}
