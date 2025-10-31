const std = @import("std");

pub const Reader = struct {
    interface: std.Io.Reader,

    pub fn initInterface() std.Io.Reader {
        return .{
            .vtable = &.{
                // .stream = Reader.stream,
                // .discard = Reader.discard,
                // .readVec = Reader.readVec,
            },
        };
    }

    pub fn init() Reader {
        return .{
            .interface = initInterface(),
        };
    }
};

pub fn reader() Reader {
    return .init();
}
