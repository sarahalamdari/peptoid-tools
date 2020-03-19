
from peptoid_tools import *
import numpy as np

a = np.array([1,0,0])
b = np.array([0,1,0])

def test_vec_align(a,b):
    r = assembler.vec_align(a,b)
    assert r is not None

r = assembler.vec_align(a,b)
r
