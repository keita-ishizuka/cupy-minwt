import cupy as cp
import numpy as np
from cupyx.profiler import benchmark

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
    k = 20
    x = cp.arange(k, dtype=cp.uint32).reshape(k)
    y = cp.array(
        [[1 if j >> i & 1 else 0 for i in range(k)] for j in range(1 << k)],
        dtype=cp.uint32,
    ).reshape(1 << k, k)
    # keepdims=True: 行列のdimを保つ（2次元の行列をreduceしたら１次元になるが、
    # それを２次元のまま表示したい場合はTrueにする）
    # axis=0: colごとにreduce
    # axis=1: rowごとにreduce
    z = gf2_matmul(x, y, axis=1, keepdims=False)
    print(x)
    print(y)
    print(z[50:70])
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
