import time
from itertools import product

import cupy
import numpy

cnt = 100


def meas_cupy(func, *args):
    stream = cupy.cuda.Stream.null
    start = stream.record()
    for i in range(cnt):
        func(*args)
    end = stream.record()
    end.synchronize()
    elapsed = cupy.cuda.get_elapsed_time(start, end) / cnt
    return elapsed


def meas_numpy(func, *args):
    start = time.time()
    for i in range(cnt):
        func(*args)
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
        cupy_elapsed = meas_cupy(func, A, B, cupy)
        numpy_elapsed = meas_numpy(func, a, b, numpy)
        meas_results.append((str(shape), numpy_elapsed, cupy_elapsed))
    for result in meas_results:
        print("{0:s}\t{1:.3f}\t{2:.3f}".format(*result))


def calc_minwt(G, p, xp):
    G = xp.asarray(G, dtype=xp.uint8)
    k, n = G.shape
    # A = []
    A = xp.array([i for i in product(range(p), repeat=k)], dtype=xp.uint8)
    xp.remainder(xp.matmul(A, G), p)
    # for a in product(range(p), repeat=k):
    #     A.append(a)
    #     if len(A) == 10**8:
    #         A = xp.asarray(A, dtype=xp.uint8)
    #         xp.remainder(xp.matmul(A, G), p)
    #         A = []
    # if A:
    #     A = xp.asarray(A, dtype=xp.uint8)
    #     xp.remainder(xp.matmul(A, G), p)


# ----------------------------------------

if __name__ == "__main__":
    G = numpy.random.randint(0, high=3, size=(20, 10))
    calc_minwt(G, 2, xp=cupy)
    print(meas_cupy(calc_minwt, G, 2, cupy))
    print(meas_numpy(calc_minwt, G, 2, numpy))

    # shapes_ = (
    #     (20, 10),
    #     (10, 5),
    # )
    # for title, func in (("LinearCode.list", "xp.remainder(xp.matmul(A, B), 2)"),):
    #     meas_numpy_cupy(title, lambda A, B, xp: eval(func), shapes_)
