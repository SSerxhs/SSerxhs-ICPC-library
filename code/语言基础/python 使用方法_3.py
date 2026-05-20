import sys, math
from collections import Counter, deque, defaultdict
from math import sin, cos, asin, acos, tan, atan, atan2, gcd, sqrt, hypot
from heapq import heapify, heappop, heappush
from itertools import permutations, product, combinations, combinations_with_replacement
from itertools import chain, accumulate, groupby, islice, tee, cycle
from functools import reduce
from queue import Queue
from bisect import bisect_left, bisect_right
from random import randint, sample
from copy import copy, deepcopy
from array import array
from types import GeneratorType
 

def bootstrap(f, stk=[]):
    def wrappedfunc(*args, **kwargs):
        if stk:
            return f(*args, **kwargs)
        else:
            to = f(*args, **kwargs)
            while True:
                if type(to) is GeneratorType:
                    stk.append(to)
                    to = next(to)
                else:
                    stk.pop()
                    if not stk:
                        break
                    to = stk[-1].send(to)
            return to

    return wrappedfunc


class FastReader:
    def __init__(self):
        self.input = sys.stdin.read().splitlines()
        self.idx = 0

    def next_line(self):
        if self.idx < len(self.input):
            line = self.input[self.idx]
            self.idx += 1
            return line
        else:
            assert False


cin = FastReader()


def input():
    global cin
    return cin.next_line()


def RI():
    return int(input())


def RF():
    return float(input())


def RII():
    return [*map(int, input().split())]

def clamp(l, h, x):
    return max(min(x, h), l)

def RM(num=2):
    v = RII()
    v[:num] = [x - 1 for x in v[:num]]
    return v

def stol(s):
    return [*map(lambda x: s.find(x), list(input()))]


def vec(*dims, val=None):
    if len(dims) == 0:
        return []
    if len(dims) == 1:
        return [val] * dims[0]
    return [vec(*dims[1:], val=val) for _ in range(dims[0])]

for _ in range(RI()):
    
