import numpy as np
from numpy.linalg import lstsq


# from https://stackoverflow.com/questions/50660389/generate-a-vector-that-is-orthogonal-to-a-set-of-other-vectors-in-any-dimension
def find_orthogonal_vec(O: np.ndarray) -> np.ndarray:
    rand_vec = np.random.rand(O.shape[0], 1)
    A = np.hstack((O, rand_vec))
    b = np.zeros(O.shape[1] + 1)
    b[-1] = 1
    res = lstsq(np.conj(A.T), b, rcond=None)[0]

    if all(np.abs(np.dot(res, col)) < 10e-9 for col in np.conj(O.T)):
        # print("Success in Stinespring-construction.")
        return res
    else:
        print("Failure in Stinespring-construction.")
        quit(-1)
