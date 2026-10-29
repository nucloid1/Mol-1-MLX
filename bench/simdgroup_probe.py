"""Can MLX's metal_kernel use simdgroup_matrix, and how is an 8x8 tile laid out across lanes?

Two things the flash-attention kernel depends on and that I would otherwise be guessing:
  1. an 8x8x8 simdgroup MMA compiles and gives the right product;
  2. which (row, col) elements each lane holds (`thread_elements()`), because the online
     softmax needs per-row max/sum and so must know which lanes share a row.

A is filled with its own flat index (A[r][c] = 8r + c), so every element a lane reports
identifies its (row, col) exactly.
"""
import numpy as np
import mlx.core as mx

from mol_mlx.guard import memory_guard

SRC = r"""
    uint lane = thread_index_in_simdgroup;
    simdgroup_float8x8 A, B, C;
    simdgroup_load(A, a_in, 8);
    simdgroup_load(B, b_in, 8);
    C = simdgroup_float8x8(0);
    simdgroup_multiply_accumulate(C, A, B, C);
    simdgroup_store(C, c_out, 8);
    auto el = A.thread_elements();
    layout[lane * 2 + 0] = el[0];
    layout[lane * 2 + 1] = el[1];
"""

kernel = mx.fast.metal_kernel(
    name="sg_probe", input_names=["a_in", "b_in"], output_names=["c_out", "layout"],
    source=SRC, header="#include <metal_simdgroup_matrix>\n")

with memory_guard(max_gib=1.0):
    a = mx.arange(64, dtype=mx.float32)
    b = mx.random.normal((64,), key=mx.random.key(0))
    c, layout = kernel(inputs=[a, b], output_shapes=[(64,), (64,)],
                       output_dtypes=[mx.float32, mx.float32],
                       grid=(32, 1, 1), threadgroup=(32, 1, 1))
    mx.eval(c, layout)
    A = np.arange(64, dtype=np.float32).reshape(8, 8)
    B = np.asarray(b).reshape(8, 8)
    err = np.abs(np.asarray(c).reshape(8, 8) - A @ B).max()
    print(f"8x8x8 MMA vs numpy: max abs err {err:.2e}  ({'OK' if err < 1e-3 else 'WRONG'})\n")

    lay = np.asarray(layout).reshape(32, 2).astype(int)
    print("lane -> (row, col0, col1)   [element value = 8*row + col]")
    rows = {}
    for lane in range(32):
        r0, c0 = divmod(lay[lane, 0], 8)
        r1, c1 = divmod(lay[lane, 1], 8)
        assert r0 == r1 and c1 == c0 + 1, f"lane {lane}: elements are not adjacent in one row"
        rows.setdefault(r0, []).append(lane)
        if lane < 12 or lane % 8 == 0:
            print(f"  lane {lane:2d} -> row {r0}, cols {c0},{c1}")
    print("\nlanes sharing each row:")
    for r in range(8):
        print(f"  row {r}: lanes {rows[r]}   (xor-differences within the group: "
              f"{sorted({a ^ b for a in rows[r] for b in rows[r] if a != b})})")
