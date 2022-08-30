import numpy as np
from cupyx.profiler import benchmark

import cupy as cp

# ReductionKernel can take multiple arguments
gf2_matmul = cp.ReductionKernel(
    # Calculate the product of two matrices x and y, regarding as matrices over GF(2).
    in_params="T x, T y",
    out_params="T z",
    map_expr="x * y",
    reduce_expr="a ^ b",
    post_map_expr="z = a",
    identity="0",
    name="gf2_matmul",
)

if __name__ == "__main__":
    x = cp.arange(10, dtype=cp.uint32).reshape(10)
    y = cp.array(
        [[0, 1, 0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 1, 1, 0, 0, 0, 0, 0]],
        dtype=cp.uint32,
    ).reshape(2, 10)
    # keepdims=True: 行列のdimを保つ（2次元の行列をreduceしたら１次元になるが、
    # それを２次元のまま表示したい場合はTrueにする）
    # axis=0: colごとにreduce
    # axis=1: rowごとにreduce
    z = gf2_matmul(x, y, axis=1, keepdims=False)
    print(x)
    print(y)
    print(z)
    print(
        benchmark(
            gf2_matmul,
            (
                x,
                y,
            ),
            n_repeat=100,
        )
    )
