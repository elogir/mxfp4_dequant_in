const std = @import("std");
const safetensors = @import("safetensors.zig");

const VEC_SIZE = 8;

const MxfpBuffer = struct {
    blocks_data: ?[]const u8 = null,
    scales_data: ?[]const u8 = null,
    blocks_info: ?safetensors.TensorInfo = null,

    fn isComplete(self: MxfpBuffer) bool {
        return self.blocks_data != null and self.scales_data != null;
    }
};

const FP4_TO_FLOAT_TABLE: [16]f32 = compute_fp4_table: {
    var table: [16]f32 = undefined;

    for (0..16) |i| {
        const fp4: u4 = @intCast(i);
        const sign = (fp4 >> 3) & 0b1;
        const exponent: i32 = @intCast((fp4 >> 1) & 0b11);
        const mantissa: f32 = @floatFromInt(fp4 & 0b1);
        const sign_val: f32 = if (sign == 1) -1.0 else 1.0;

        if (exponent == 0) {
            // La formule exacte est: sign_val * 2^(1 - bias) * (0 + 2^(-m) x M)
            // Or bias = 1 et m = 1 donc: sign_val * 1 * 2^(-1) * M = sign_val * 0.5 * M
            table[i] = sign_val * 0.5 * mantissa;
        } else {
            // Normal: sign_val * 2^(E-bias) * (1 + 2^(-m) * M)
            // Or bias = 1 et m = 1 donc: sign_val * 2^(E-1) * (1 + 2^(-1) * M) = sign_val * 2(E-1) * (1 + M/2)
            const mant_val = 1.0 + mantissa / 2.0;
            table[i] = std.math.scalbn(sign_val, exponent - 1) * mant_val;
        }
    }

    break :compute_fp4_table table;
};

inline fn fp4ToFloat(fp4: u4) f32 {
    return FP4_TO_FLOAT_TABLE[fp4];
}

inline fn e8m0ToFloat(e8m0: u8) f32 {
    // NaN selon spec
    if (e8m0 == 0b11111111) return std.math.nan(f32);

    // Conventional biased Float32 exponent: 2^(e8m0 - 127)
    const exponent: i32 = @as(i32, @intCast(e8m0)) - 127;
    return std.math.scalbn(@as(f32, 1.0), exponent);
}

inline fn floatToBF16(value: f32) u16 {
    const bits: u32 = @bitCast(value);

    return @intCast(bits >> 16);
}

fn dequantizeMxfp4(allocator: std.mem.Allocator, tensor_buffer: MxfpBuffer, output_writer: *std.Io.Writer) !void {
    const blocks_info = tensor_buffer.blocks_info.?;
    const blocks_data = tensor_buffer.blocks_data.?;
    const scales_data = tensor_buffer.scales_data.?;

    const n_experts = blocks_info.shape[0];
    const out_features = blocks_info.shape[1];
    const n_blocks = blocks_info.shape[2];
    const blk_size = blocks_info.shape[3];
    const in_features = n_blocks * blk_size * 2;

    // Buffer pour un expert a la fois
    const expert_buffer_a_volonte = try allocator.alloc(u16, in_features * out_features);
    defer allocator.free(expert_buffer_a_volonte);

    for (0..n_experts) |expert_idx| {
        const scales_offset = expert_idx * out_features * n_blocks;
        const blocks_offset = expert_idx * out_features * n_blocks * blk_size;

        const scales_buf = scales_data[scales_offset..][0 .. out_features * n_blocks];
        const blocks_buf = blocks_data[blocks_offset..][0 .. out_features * n_blocks * blk_size];

        for (0..out_features) |out_idx| {
            for (0..n_blocks) |block_idx| {
                // Get scale
                const scale_offset = out_idx * n_blocks + block_idx;
                const scale_byte = scales_buf[scale_offset];
                const scale = e8m0ToFloat(scale_byte);
                const scale_vec: @Vector(VEC_SIZE, f32) = @splat(scale);

                // Get block data
                const block_offset = out_idx * n_blocks * blk_size + block_idx * blk_size;
                const block = blocks_buf[block_offset..][0..blk_size];

                var byte_idx: usize = 0;
                while (byte_idx + 4 <= blk_size) : (byte_idx += 4) {
                    // Charger 4 bytes = 8 valeurs fp4
                    var fp4_values: @Vector(VEC_SIZE, u4) = undefined; // ex: vecsize = 8 donc 8x4=32
                    inline for (0..4) |i| {
                        const byte = block[byte_idx + i];

                        const fp4_low: u4 = @intCast(byte & 0xF);
                        const fp4_high: u4 = @intCast((byte >> 4) & 0xF);

                        fp4_values[i * 2] = fp4_low;
                        fp4_values[i * 2 + 1] = fp4_high;
                    }

                    // Convertir les 8 fp4 en f32
                    var fp4_floats: @Vector(VEC_SIZE, f32) = undefined;
                    inline for (0..VEC_SIZE) |i| {
                        fp4_floats[i] = fp4ToFloat(fp4_values[i]); // Je suppose que vu que la func est inline ca va juste charger directement?
                    }

                    // Appliquer le scale vectorised
                    const scaled_vec = fp4_floats * scale_vec;

                    // Conversion bf16
                    inline for (0..VEC_SIZE) |i| {
                        const bf16_value = floatToBF16(scaled_vec[i]); // Pareil que au dessus

                        const in_idx = block_idx * blk_size * 2 + byte_idx * 2 + i;
                        expert_buffer_a_volonte[in_idx * out_features + out_idx] = bf16_value;
                    }
                }
            }
        }

        const output_bytes: []u8 = std.mem.sliceAsBytes(expert_buffer_a_volonte);
        try output_writer.writeAll(output_bytes);
    }
}

/// Litteralement le meme principe que pour parse le header avec les pair
/// Si on fait exactement la meme chose que ce qu'on a fait avec le header, on devrait avoir le meme resultat
pub fn processTensors(
    allocator: std.mem.Allocator,
    old_tensors: []safetensors.TensorInfo,
    input_reader: *std.Io.Reader,
    output_writer: *std.Io.Writer,
) !void {
    var mxfp_buffers_pair: std.StringHashMap(MxfpBuffer) = .init(allocator);
    defer {
        var iter = mxfp_buffers_pair.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.blocks_data) |data| allocator.free(data);
            if (entry.value_ptr.scales_data) |data| allocator.free(data);
        }
        mxfp_buffers_pair.deinit();
    }

    for (old_tensors) |tensor| {
        const tensor_data = try allocator.alloc(u8, tensor.size());
        errdefer allocator.free(tensor_data);

        try input_reader.readSliceAll(tensor_data); // On lit le tenseur en entier (normalement la taille dedvrait aller dans la ram?)

        if (tensor.is_mxfp4_blocks or tensor.is_mxfp4_scales) {
            const entry = try mxfp_buffers_pair.getOrPut(tensor.base_name);
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
                std.debug.print("Dequantizing MXFP4 pair: {s}\n", .{tensor.base_name});

                const blocks = entry.value_ptr.blocks_data.?;
                const scales = entry.value_ptr.scales_data.?;

                try dequantizeMxfp4(allocator, entry.value_ptr.*, output_writer);

                allocator.free(blocks);
                allocator.free(scales);
                _ = mxfp_buffers_pair.remove(tensor.base_name);
            }
        } else {
            std.debug.print("Copying regular tensor: {s}\n", .{tensor.base_name});

            try output_writer.writeAll(tensor_data);
            allocator.free(tensor_data);
        }
    }
}
