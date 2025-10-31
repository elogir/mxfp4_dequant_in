const std = @import("std");
const json = std.json;
const safetensors = @import("safetensors.zig");
const mxfp4 = @import("mxfp4.zig");
const dequant = @import("dequant.zig");

pub fn dequant_safetensors(allocator: std.mem.Allocator, arena: std.mem.Allocator, tensor_reader: *std.Io.Reader, buffer: []u8) !dequant.Reader {
    return dequant.Reader.init(allocator, arena, tensor_reader, buffer);
}
