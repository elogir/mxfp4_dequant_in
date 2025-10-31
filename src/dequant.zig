const std = @import("std");
const safetensors = @import("safetensors.zig");
const mxfp4 = @import("mxfp4.zig");
const json = std.json;

const VEC_SIZE = 16;
const VEC_SIZE_FP4 = VEC_SIZE / 2;

/// Quasi la meme que TensorInfo mais avec la data
/// (normalement free quasi juste apres vu que les tenseurs sont censes etre adjacents)
const MxfpBuffer = struct {
    blocks_data: ?[]const u8 = null,
    scales_data: ?[]const u8 = null,
    blocks_info: ?safetensors.TensorInfo = null,

    fn isComplete(self: MxfpBuffer) bool {
        return self.blocks_data != null and self.scales_data != null;
    }
};

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
    mxfp_buffers_pair: std.StringHashMap(MxfpBuffer),
    output_buff: []u8,
    tensor_index: usize = 0,

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
            .mxfp_buffers_pair = std.StringHashMap(MxfpBuffer).init(allocator),
            .output_buff = &[_]u8{},
            .interface = initInterface(buffer),
        };
    }

    pub fn deinit(r: *Reader) void {
        r.allocator.free(r.new_header_json_bytes);
        // Le header est dans l'arena

        var iter = r.mxfp_buffers_pair.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.blocks_data) |data| r.allocator.free(data);
            if (entry.value_ptr.scales_data) |data| r.allocator.free(data);
        }
        r.mxfp_buffers_pair.deinit();
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
                    r.send_offset = 0;
                }

                return to_send.len;
            },

            .tensor_dequant => {
                if (r.send_offset < r.output_buff.len) {
                    const remaining = r.output_buff[r.send_offset..];
                    const to_send = limit.slice(remaining);

                    try w.writeAll(to_send);

                    r.send_offset += to_send.len;
                    return to_send.len;
                }

                r.allocator.free(r.output_buff); // On devrait avoir une autre facon plutot que de alloc a chaque fois
                r.output_buff = &[_]u8{};

                if (r.tensor_index >= r.old_header.len) {
                    r.incrementState();
                    return 0;
                }

                const tensor = r.old_header[r.tensor_index];
                r.tensor_index += 1;

                if (tensor.is_mxfp4_blocks or tensor.is_mxfp4_scales) {
                    // handle mxfp4 tensor
                    std.debug.print("Dequantizing MXFP4 pair: {s}\n", .{tensor.base_name});
                    r.processTensor(tensor) catch {
                        return error.ReadFailed;
                    };
                } else {
                    // Handle regular tensor (copy)
                    std.debug.print("Copying regular tensor: {s}\n", .{tensor.base_name});
                    r.output_buff = r.allocator.alloc(u8, tensor.size()) catch {
                        return error.ReadFailed;
                    };

                    r.input_reader.readSliceAll(r.output_buff[0..tensor.size()]) catch {
                        return error.ReadFailed;
                    };

                    r.send_offset = 0;
                }

                return stream(io_reader, w, limit);
            },

            .finished => {
                return error.EndOfStream;
            },
        }
    }

    /// Litteralement le meme principe que pour parse le header avec les paires
    /// Si on fait exactement la meme chose que ce qu'on a fait avec le header, on devrait avoir le meme resultat
    fn processTensor(r: *Reader, tensor: safetensors.TensorInfo) !void {
        const tensor_data = try r.allocator.alloc(u8, tensor.size());
        errdefer r.allocator.free(tensor_data);

        try r.input_reader.readSliceAll(tensor_data); // On lit le tenseur en entier (normalement la taille dedvrait aller dans la ram?)

        const entry = try r.mxfp_buffers_pair.getOrPut(tensor.base_name);
        if (!entry.found_existing) {
            entry.value_ptr.* = MxfpBuffer{};
        }

        if (tensor.is_mxfp4_blocks) {
            entry.value_ptr.blocks_data = tensor_data;
            entry.value_ptr.blocks_info = tensor;
        } else {
            entry.value_ptr.scales_data = tensor_data;
        }

        if (entry.value_ptr.isComplete()) {
            const blocks = entry.value_ptr.blocks_data.?;
            const scales = entry.value_ptr.scales_data.?;

            try r.dequantizeMxfp4(entry.value_ptr.*);

            r.allocator.free(blocks);
            r.allocator.free(scales);
            _ = r.mxfp_buffers_pair.remove(tensor.base_name);
        }
        // If incomplete, do nothing (on attend l'autre paire)
    }

    fn dequantizeMxfp4(r: *Reader, tensor_buffer: MxfpBuffer) !void {
        const blocks_info = tensor_buffer.blocks_info.?;
        const blocks_data = tensor_buffer.blocks_data.?;
        const scales_data = tensor_buffer.scales_data.?;

        const n_experts = blocks_info.shape[0];
        const out_features = blocks_info.shape[1];
        const n_blocks = blocks_info.shape[2];
        const blk_size = blocks_info.shape[3];
        const in_features = n_blocks * blk_size * 2;

        // Buffer pour un expert a la fois
        // const expert_buffer_a_volonte = try r.allocator.alloc(u16, in_features * out_features);
        // defer r.allocator.free(expert_buffer_a_volonte);

        const siz = n_experts * out_features * in_features * 2;
        r.output_buff = try r.allocator.alloc(u8, siz);
        const buff: []align(1) u16 = std.mem.bytesAsSlice(u16, r.output_buff[0..siz]); // hack

        for (0..n_experts) |expert_idx| {
            const expert_offset = expert_idx * out_features * in_features;
            const scales_offset = expert_idx * out_features * n_blocks;
            const blocks_offset = expert_idx * out_features * n_blocks * blk_size;

            const scales_buf = scales_data[scales_offset..][0 .. out_features * n_blocks];
            const blocks_buf = blocks_data[blocks_offset..][0 .. out_features * n_blocks * blk_size];

            for (0..out_features) |out_idx| {
                for (0..n_blocks) |block_idx| {
                    // Get scale
                    const scale_offset = out_idx * n_blocks + block_idx;
                    const scale_byte = scales_buf[scale_offset];
                    const scale = mxfp4.e8m0ToFloat(scale_byte);
                    const scale_vec: @Vector(VEC_SIZE, f32) = @splat(scale);

                    // Get block data
                    const block_offset = out_idx * n_blocks * blk_size + block_idx * blk_size;
                    const block = blocks_buf[block_offset..][0..blk_size];

                    var byte_idx: usize = 0;
                    std.debug.assert(blk_size % VEC_SIZE_FP4 == 0); // Sinon faut traiter la tail sans vec
                    while (byte_idx + VEC_SIZE_FP4 <= blk_size) : (byte_idx += VEC_SIZE_FP4) {
                        // Charger VEC_SIZE/2 bytes = VEC_SIZE valeurs fp4
                        var fp4_values: @Vector(VEC_SIZE, u4) = undefined;
                        inline for (0..VEC_SIZE_FP4) |i| {
                            const byte = block[byte_idx + i];

                            const fp4_low: u4 = @intCast(byte & 0xF);
                            const fp4_high: u4 = @intCast((byte >> 4) & 0xF);

                            fp4_values[i * 2] = fp4_low;
                            fp4_values[i * 2 + 1] = fp4_high;
                        }

                        // Convertir les 8 fp4 en f32 (@shuffle? mais on a pas de values au comptime?)
                        var fp4_floats: @Vector(VEC_SIZE, f32) = undefined;
                        inline for (0..VEC_SIZE) |i| {
                            fp4_floats[i] = mxfp4.fp4ToFloat(fp4_values[i]); // Je suppose que vu que la func est inline ca va juste charger directement?
                        }

                        // Appliquer le scale vectorised
                        const scaled_vec = fp4_floats * scale_vec;

                        // Conversion bf16
                        inline for (0..VEC_SIZE) |i| {
                            const bf16_value = mxfp4.floatToBF16(scaled_vec[i]); // Pareil que au dessus

                            const in_idx = block_idx * blk_size * 2 + byte_idx * 2 + i;
                            buff[expert_offset + in_idx * out_features + out_idx] = bf16_value;
                        }
                    }
                }
            }
        }

        r.send_offset = 0;
    }
};
