from importlib.resources import files
import numpy as np
from scipy.special import gammaln

def get_project_root():
    return files("transformer_reasoning")._paths[0].parent.parent

def get_src_root():
    return files("transformer_reasoning")._paths[0]

def log_double_factorial(n):
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == -1 or n == 0:
        return 0.0  # By definition, (-1)!! = 1 and 0!! = 1
    if n % 2 == 0:  # n is even
        k = n // 2
        return k * np.log(2) + gammaln(k + 1)
    else:  # n is odd
        k = (n + 1) // 2
        return gammaln(n + 1) - (k * np.log(2) + gammaln(k + 1))