const std = @import("std");
const safetensors = @import("safetensors.zig");
const mxfp4 = @import("mxfp4.zig");
const json = std.json;

pub const Reader = struct {
    interface: std.Io.Reader,

    input_reader: *std.Io.Reader,
    allocator: std.mem.Allocator,
    header_arena: std.mem.Allocator,

    // Everything header related
    old_header: []safetensors.TensorInfo,
    new_header: []safetensors.TensorInfo,
    new_header_json_bytes: []u8,
    new_header_len: u64,
    header_len_sent: bool = false, // tracking header len sent
    header_bytes_pos: usize = 0, // tracking header bytes sent

    // Everything tensor related
    mxfp_buffers_pair: std.StringHashMap(mxfp4.MxfpBuffer),
    output_buff: []u8,
    tensor_idnex: usize = 0,

    send_state: SendState = .header_length,
    send_offset: usize = 0,

    const SendState = enum {
        header_length,
        header_content,
        tensor_dequant,
        finished,
    };

    inline fn incrementState(self: *Reader) void {
        self.send_state = @enumFromInt(@intFromEnum(self.send_state) + 1);
    }

    pub fn initInterface(buffer: []u8) std.Io.Reader {
        return .{
            .vtable = &.{
                .stream = Reader.stream,
            },
            .buffer = buffer,
            .seek = 0,
            .end = 0,
        };
    }

    pub fn init(
        allocator: std.mem.Allocator,
        header_arena: std.mem.Allocator,
        input_reader: *std.Io.Reader,
        buffer: []u8,
    ) !Reader {
        const header_result = try safetensors.parseHeader(header_arena, input_reader);
        const new_header_json_bytes = try json.Stringify.valueAlloc(
            allocator,
            json.Value{ .object = header_result.new_json.* },
            .{ .whitespace = .minified },
        );

        return .{
            .allocator = allocator,
            .header_arena = header_arena,
            .input_reader = input_reader,
            .old_header = header_result.old_header,
            .new_header = header_result.new_header,
            .new_header_json_bytes = new_header_json_bytes,
            .new_header_len = @intCast(new_header_json_bytes.len),
            .mxfp_buffers_pair = std.StringHashMap(mxfp4.MxfpBuffer).init(allocator),
            .interface = initInterface(buffer),
        };
    }

    pub fn deinit(r: *Reader) void {
        r.allocator.free(r.new_header_json_bytes);
        // Le header est dans l'arena

        defer {
            var iter = r.mxfp_buffers_pair.iterator();
            while (iter.next()) |entry| {
                if (entry.value_ptr.blocks_data) |data| r.allocator.free(data);
                if (entry.value_ptr.scales_data) |data| r.allocator.free(data);
            }
            r.mxfp_buffers_pair.deinit();
        }
    }

    fn stream(
        io_reader: *std.Io.Reader,
        w: *std.Io.Writer,
        limit: std.Io.Limit,
    ) std.Io.Reader.StreamError!usize {
        const r: *Reader = @alignCast(@fieldParentPtr("interface", io_reader));

        switch (r.send_state) {
            .header_length => { // Etape 1 : send la len du nouveau header
                var len_bytes: [8]u8 = undefined;
                std.mem.writeInt(u64, &len_bytes, r.new_header_len, .little);

                const remaining = len_bytes[r.send_offset..];
                const to_send = limit.slice(remaining);
                try w.writeAll(to_send);
                r.send_offset += to_send.len;

                if (r.send_offset == 8) {
                    r.incrementState();
                    r.send_offset = 0;
                }

                return to_send.len;
            },

            .header_content => { // Etape 2 : le nouveau header
                const remaining = r.new_header_json_bytes[r.send_offset..];
                const to_send = limit.slice(remaining);
                try w.writeAll(to_send);
                r.send_offset += to_send.len;

                if (r.send_offset == r.new_header_json_bytes.len) {
                    r.incrementState();
                }

                return to_send.len;
            },

            .tensor_dequant => {
                r.incrementState();
                return 0;
            },

            .finished => {
                return error.EndOfStream;
            },
        }
    }
};
