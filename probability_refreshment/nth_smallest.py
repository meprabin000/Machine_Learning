import numpy as np

def nth_smallest(data, top, bottom, left, right, n):
    return np.sort(data[top:bottom+1, left:right+1].flatten())[n-1]