const std = @import("std");
const json = std.json;

fn parse_header(allocator: std.mem.Allocator, tensor_reader: *std.Io.Reader) !void {

    // TEMP
    var buff_a_volonter: [0xFFFF]u8 = undefined;
    const output_file = try std.fs.cwd().createFile("TEST_OUT_HEADER.json", .{});
    const output_writer = output_file.writer(&buff_a_volonter);
    const output = &output_writer.interface;
    defer output_file.close();

    // TEMP

    const header_size: usize = @intCast(try tensor_reader.takeInt(u64, .little));
    const header_data = try allocator.alloc(u8, header_size);
    defer allocator.free(header_data);

    try tensor_reader.readSliceAll(header_data);

    const parsed = try json.parseFromSlice(json.Value, allocator, header_data, .{});
    defer parsed.deinit();

    const header = parsed.value.object;
    var iter = header.iterator();
    while (iter.next()) |entry| {
        const tensor_name = entry.key_ptr.*;
        const tensor_info = entry.value_ptr.*;

        if (std.mem.eql(u8, tensor_name, "__metadata__")) continue;
        std.debug.print("name: {s} | info: {}\n", .{ tensor_name, @TypeOf(tensor_info) });

        const dtype = tensor_info.object.get("dtype").?.string;
        const shape = tensor_info.object.get("shape").?.array;
        // const data_offsets = tensor_info.object.get("data_offsets").?.array;

        std.debug.print("\nTensor: {s}\n", .{tensor_name});
        std.debug.print("  dtype: {s}\n", .{dtype});
        std.debug.print("  shape: [", .{});
        for (shape.items, 0..) |dim, i| {
            if (i > 0) std.debug.print(", ", .{});
            std.debug.print("{}", .{dim.integer});
        }
        std.debug.print("]\n", .{});
    }
}

pub fn dequant_tensor(allocator: std.mem.Allocator, tensor_reader: *std.Io.Reader) !void {
    try parse_header(allocator, tensor_reader);
}
