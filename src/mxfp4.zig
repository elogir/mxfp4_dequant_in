const std = @import("std");

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
