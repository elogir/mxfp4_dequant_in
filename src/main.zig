const std = @import("std");
const mxfp4Loader = @import("mxfp4Loader");

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    var header_arena: std.heap.ArenaAllocator = .init(allocator);
    defer header_arena.deinit();
    const arena = header_arena.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len != 3) {
        std.debug.print("Usage: {s} <input_safetensors> <output_safetensors>\n", .{args[0]});
        return error.InvalidArguments;
    }

    const safetensors_path = args[1];
    const output_path = args[2];

    const safetensors_file = try std.fs.cwd().openFile(safetensors_path, .{});
    defer safetensors_file.close();

    var safetensors_buf: [4096]u8 = undefined;
    var safetensors_reader = safetensors_file.reader(&safetensors_buf);

    const output_file = try std.fs.cwd().createFile(output_path, .{});
    defer output_file.close();

    var file_buffer: [0xFFFF]u8 = undefined;
    var file_writer = output_file.writer(&file_buffer);

    var dequant_buffer: [8192]u8 = undefined;
    var dequant_reader = try mxfp4Loader.dequant_safetensors(
        allocator,
        arena, // Arena pour pouvoir utiliser le json leaky
        &safetensors_reader.interface,
        &dequant_buffer,
    );
    defer dequant_reader.deinit();

    _ = try dequant_reader.interface.streamRemaining(&file_writer.interface);

    try file_writer.interface.flush();
}
