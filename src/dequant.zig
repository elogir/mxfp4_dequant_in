const std = @import("std");

pub const Reader = struct {
    interface: std.Io.Reader,

    input_reader: *std.Io.Reader,
    allocator: std.mem.Allocator,
    header_arena: std.mem.Allocator,

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
    ) Reader {
        return .{
            .allocator = allocator,
            .header_arena = header_arena,
            .input_reader = input_reader,
            .interface = initInterface(buffer),
        };
    }

    pub fn deinit(_: *Reader) void {}

    fn stream(
        io_reader: *std.Io.Reader,
        w: *std.Io.Writer,
        limit: std.Io.Limit,
    ) std.Io.Reader.StreamError!usize {
        _ = io_reader;
        _ = w;
        _ = limit;

        return 0;
    }
};
