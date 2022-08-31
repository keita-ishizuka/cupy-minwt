# 以下のGistから拝借
# https://gist.github.com/magnium/cf96160d248a79f9463439695a7748e8
# 出力結果は，cupy_vs_numpy.outにある

import time

import cupy
import numpy

cnt = 100
N = 10


def meas_cupy(func, operand):
    stream = cupy.cuda.Stream.null
    start = stream.record()
    for i in range(cnt):
        func(operand, xp=cupy)
    end = stream.record()
    end.synchronize()
    elapsed = cupy.cuda.get_elapsed_time(start, end) / cnt
    return elapsed


def meas_numpy(func, operand):
    start = time.time()
    for i in range(cnt):
        func(operand, xp=numpy)
    end = time.time()
    elapsed = (end - start) * 1000 / cnt
    return elapsed


def meas_numpy_cupy(title, func, shapes):
    meas_results = []
    print()
    print("===", title, "===")
    print("Array size\tNumpy\tCupy")
    for shape in shapes:
        A = cupy.random.rand(*shape)
        a = A.get()
        cupy_elapsed = meas_cupy(func, A)
        numpy_elapsed = meas_numpy(func, a)
        meas_results.append((a.size, numpy_elapsed, cupy_elapsed))
    for result in meas_results:
        print("{0:10d}\t{1:.3f}\t{2:.3f}".format(*result))


# ----------------------------------------

shapes_ = ((N, N), (N, N, N, N), (N, N, N, N, N, N), (N, N, N, N, N, N, N))

for title, func in (
    ("Array add", "A+A"),
    ("Array sub", "A-A"),
    ("Array sum", "A.sum()"),
    ("Array argmax", "A.argmax()"),
    ("Array sort", "xp.sort(A)"),
):
    meas_numpy_cupy(title, lambda A, xp: eval(func), shapes_)

shapes_ = ((N, N), (N, N, N, N), (N, N, N, N, N, N))

for title, func in (
    ("Array tensordot", "xp.tensordot(A, A)"),
    ("Array matmul", "xp.matmul(A, A)"),
    ("Array einsum", "xp.einsum('...i, ...j->...ij', A, A)"),
    ("Array transpose", "xp.moveaxis(A, 0, -1)"),
    ("Array sin", "xp.sin(A)"),
):
    meas_numpy_cupy(title, lambda A, xp: eval(func), shapes_)

shapes_ = ((N, N), (N * N, N * N), (N * N * N, N * N * N))

for title, func in (
    ("Array eigenvalue", "xp.linalg.eigh(A)"),
    ("Array inv", "xp.linalg.inv(A)"),
):
    meas_numpy_cupy(title, lambda A, xp: eval(func), shapes_)
