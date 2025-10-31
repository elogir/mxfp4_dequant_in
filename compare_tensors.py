#!/usr/bin/env python3
"""
Compare tensors between two SafeTensors files.
Checks that all tensors match in name, shape, dtype, and values.
"""

import sys
from pathlib import Path
from safetensors import safe_open
import numpy as np


def bfloat16_to_float32(bf16_bytes):
    """Convert bfloat16 bytes to float32 numpy array."""
    # BF16 is 16 bits: 1 sign bit, 8 exponent bits, 7 mantissa bits
    # We can convert to FP32 by adding 16 zero bits to the right
    bf16_array = np.frombuffer(bf16_bytes, dtype=np.uint16)
    fp32_bits = bf16_array.astype(np.uint32) << 16
    fp32_array = fp32_bits.view(np.float32)
    return fp32_array


def load_tensors(filepath):
    """Load all tensors from a SafeTensors file into a dictionary."""
    import json
    import struct

    tensors = {}

    # First try with numpy framework for standard dtypes
    try:
        with safe_open(filepath, framework="numpy") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        return tensors
    except TypeError as e:
        if "bfloat16" not in str(e):
            raise

    # If we hit bfloat16, read manually
    with open(filepath, "rb") as f:
        # Read header size (8 bytes, little endian u64)
        header_size_bytes = f.read(8)
        header_size = struct.unpack("<Q", header_size_bytes)[0]

        # Read header JSON
        header_bytes = f.read(header_size)
        header = json.loads(header_bytes.decode("utf-8"))

        # Process each tensor
        for key, info in header.items():
            if key == "__metadata__":
                continue

            dtype = info["dtype"]
            shape = info["shape"]
            data_offsets = info["data_offsets"]

            # Calculate position in file (header_size + 8 for the size prefix)
            start = 8 + header_size + data_offsets[0]
            end = 8 + header_size + data_offsets[1]
            size = end - start

            # Seek and read raw data
            f.seek(start)
            raw_data = f.read(size)

            # Convert based on dtype
            if dtype == "BF16":
                tensor = bfloat16_to_float32(raw_data)
                tensor = tensor.reshape(shape)
            elif dtype == "F32":
                tensor = np.frombuffer(raw_data, dtype=np.float32).reshape(shape)
            elif dtype == "F16":
                tensor = np.frombuffer(raw_data, dtype=np.float16).reshape(shape).astype(np.float32)
            elif dtype == "I32":
                tensor = np.frombuffer(raw_data, dtype=np.int32).reshape(shape)
            elif dtype == "I64":
                tensor = np.frombuffer(raw_data, dtype=np.int64).reshape(shape)
            else:
                print(f"Warning: Unknown dtype {dtype} for tensor {key}, skipping")
                continue

            tensors[key] = tensor

    return tensors


def compare_tensors(tensor1, tensor2, name, rtol=1e-5, atol=1e-8):
    """Compare two tensors and return comparison results."""
    results = {
        "name": name,
        "match": True,
        "shape_match": True,
        "dtype_match": True,
        "values_match": True,
        "messages": []
    }

    # Compare shapes
    if tensor1.shape != tensor2.shape:
        results["match"] = False
        results["shape_match"] = False
        results["messages"].append(
            f"Shape mismatch: {tensor1.shape} vs {tensor2.shape}"
        )

    # Compare dtypes
    if tensor1.dtype != tensor2.dtype:
        results["match"] = False
        results["dtype_match"] = False
        results["messages"].append(
            f"Dtype mismatch: {tensor1.dtype} vs {tensor2.dtype}"
        )

    # Compare values (if shapes match)
    if results["shape_match"]:
        if not np.allclose(tensor1, tensor2, rtol=rtol, atol=atol):
            results["match"] = False
            results["values_match"] = False

            # Calculate statistics about the difference
            diff = np.abs(tensor1 - tensor2)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)

            # Find where the maximum difference occurs
            max_idx = np.unravel_index(np.argmax(diff), diff.shape)

            results["messages"].append(
                f"Values differ: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}"
            )
            results["messages"].append(
                f"Max diff at index {max_idx}: {tensor1[max_idx]} vs {tensor2[max_idx]}"
            )

            # Count how many values are different
            num_different = np.sum(~np.isclose(tensor1, tensor2, rtol=rtol, atol=atol))
            total_elements = tensor1.size
            results["messages"].append(
                f"Different elements: {num_different}/{total_elements} "
                f"({100.0 * num_different / total_elements:.2f}%)"
            )

    return results


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_tensors.py <file1.safetensors> <file2.safetensors>")
        print("\nExample:")
        print("  python compare_tensors.py test_models/tiny_gpt_oss/test_output_model.safetensors output.safetensors")
        sys.exit(1)

    file1_path = Path(sys.argv[1])
    file2_path = Path(sys.argv[2])

    # Check files exist
    if not file1_path.exists():
        print(f"Error: File not found: {file1_path}")
        sys.exit(1)
    if not file2_path.exists():
        print(f"Error: File not found: {file2_path}")
        sys.exit(1)

    print(f"Comparing tensors:")
    print(f"  File 1: {file1_path}")
    print(f"  File 2: {file2_path}")
    print()

    # Load tensors from both files
    print("Loading tensors from file 1...")
    tensors1 = load_tensors(file1_path)
    print(f"  Loaded {len(tensors1)} tensors")

    print("Loading tensors from file 2...")
    tensors2 = load_tensors(file2_path)
    print(f"  Loaded {len(tensors2)} tensors")
    print()

    # Get all tensor names
    names1 = set(tensors1.keys())
    names2 = set(tensors2.keys())

    # Find common and unique tensors
    common_names = names1 & names2
    only_in_file1 = names1 - names2
    only_in_file2 = names2 - names1

    print(f"Tensor name comparison:")
    print(f"  Common tensors: {len(common_names)}")
    print(f"  Only in file 1: {len(only_in_file1)}")
    print(f"  Only in file 2: {len(only_in_file2)}")
    print()

    if only_in_file1:
        print("Tensors only in file 1:")
        for name in sorted(only_in_file1):
            print(f"  - {name}")
        print()

    if only_in_file2:
        print("Tensors only in file 2:")
        for name in sorted(only_in_file2):
            print(f"  - {name}")
        print()

    # Compare common tensors
    if common_names:
        print(f"Comparing {len(common_names)} common tensors:")
        print("-" * 80)

        all_match = True
        results_list = []

        for name in sorted(common_names):
            result = compare_tensors(tensors1[name], tensors2[name], name)
            results_list.append(result)

            if result["match"]:
                print(f"✓ {name}: MATCH (shape={tensors1[name].shape}, dtype={tensors1[name].dtype})")
            else:
                all_match = False
                print(f"✗ {name}: MISMATCH")
                for msg in result["messages"]:
                    print(f"    {msg}")

        print("-" * 80)
        print()

        # Summary
        num_matching = sum(1 for r in results_list if r["match"])
        num_total = len(results_list)

        print("Summary:")
        print(f"  Matching tensors: {num_matching}/{num_total}")
        print(f"  Mismatching tensors: {num_total - num_matching}/{num_total}")
        print()

        if all_match and len(only_in_file1) == 0 and len(only_in_file2) == 0:
            print("✓ All tensors match perfectly!")
            return 0
        else:
            print("✗ Some tensors differ or are missing")
            return 1
    else:
        print("No common tensors to compare!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
