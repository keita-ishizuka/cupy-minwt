import datetime

import cupy as xp
import numpy as np


def test(xp):
    a = xp.arange(10**6).reshape(1000, -1)
    return a.T * 2


test(np)
t1 = datetime.datetime.now()
for i in range(1000):
    test(np)
t2 = datetime.datetime.now()
print(t2 - t1)

xp.cuda.set_allocator(xp.cuda.MemoryPool().malloc)
test(xp)
t1 = datetime.datetime.now()
for i in range(1000):
    test(xp)
t2 = datetime.datetime.now()
print(t2 - t1)
