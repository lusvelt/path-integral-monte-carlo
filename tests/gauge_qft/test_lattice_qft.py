import numpy as np
from lqfn.gauge_qft.lattice_qcd import *


def test_factorial():
    assert factorial(0) == 1
    assert factorial(1) == 1
    assert factorial(2) == 2
    assert factorial(3) == 6
    assert factorial(4) == 24
    assert factorial(5) == 120
    assert factorial(6) == 720


def test_generate_random_SU3_update_matrix():
    for _ in range(20):
        M = generate_random_SU3_update_matrix(eps=0.24, taylor_order=10)
        M_dag = M.conj().T
        assert np.isclose(np.linalg.det(M), 1.0)
        assert np.isclose(M @ M_dag, np.identity(3)).all()


def test_generate_update_matrices_set():
    N = 50
    Ms = generate_update_matrices_set(N=N, eps=0.24)
    for i in range(N):
        M = Ms[i]
        M_dag = Ms[N + i]
        assert np.isclose(M.conj().T, M_dag).all()
        assert np.isclose(np.linalg.det(M), 1.0)
        print(M @ M_dag)
        assert np.isclose(M @ M_dag, np.identity(3)).all()


def test_decode_index():
    assert decode_index(15, 4, 2) == np.array([3, 3])
    assert decode_index(2431, 8, 4) == np.array([4, 5, 7, 7])


def test_encode_index():
    assert encode_index(np.array([4, 5, 7, 7]), 8, 4)
    assert encode_index(np.array([3, 3]), 4, 2)
