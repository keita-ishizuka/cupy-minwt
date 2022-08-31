import time
from itertools import product

import cupy
import numpy

cnt = 100


def meas_cupy(func, A, B):
    stream = cupy.cuda.Stream.null
    start = stream.record()
    for i in range(cnt):
        func(A, B, xp=cupy)
    end = stream.record()
    end.synchronize()
    elapsed = cupy.cuda.get_elapsed_time(start, end) / cnt
    return elapsed


def meas_numpy(func, A, B):
    start = time.time()
    for i in range(cnt):
        func(A, B, xp=numpy)
    end = time.time()
    elapsed = (end - start) * 1000 / cnt
    return elapsed


def meas_numpy_cupy(title, func, shapes):
    meas_results = []
    print()
    print("===", title, "===")
    print("Array size\tNumpy\tCupy")
    for shape in shapes:
        A = cupy.array(
            [i for i in product(range(2), repeat=shape[0])], dtype=cupy.uint8
        )
        B = cupy.random.randint(0, high=3, size=shape)
        a = A.get()
        b = B.get()
        cupy_elapsed = meas_cupy(func, A, B)
        numpy_elapsed = meas_numpy(func, a, b)
        meas_results.append((str(shape), numpy_elapsed, cupy_elapsed))
    for result in meas_results:
        print("{0:s}\t{1:.3f}\t{2:.3f}".format(*result))


# ----------------------------------------

if __name__ == "__main__":
    shapes_ = (
        (20, 10),
        (10, 5),
    )
    for title, func in (("LinearCode.list", "xp.remainder(xp.matmul(A, B), 2)"),):
        meas_numpy_cupy(title, lambda A, B, xp: eval(func), shapes_)
