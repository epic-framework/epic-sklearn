import pytest
import numpy as np
from scipy import sparse as sp

from epic.sklearn.metrics.leven import Levenshtein


norm = pytest.mark.parametrize("normalize", [True, False])
EXAMPLES = {
    ('assaf', 'asdf'): 2,
}


@norm
def test_distance(normalize):
    leven = Levenshtein(normalize=normalize)
    for (x, y), d in EXAMPLES.items():
        if normalize:
            d /= max(len(x), len(y))
        assert leven.distance(x, y) == d
        assert leven.distance(x.encode(), y.encode()) == d
        with pytest.raises(ValueError):
            leven.distance(x.encode(), y)


@norm
def test_pairwise(normalize):
    leven = Levenshtein(normalize=normalize)
    for (x, y), d in EXAMPLES.items():
        if normalize:
            d /= max(len(x), len(y))
        p = leven.pairwise((x, y))
        assert isinstance(p, np.ndarray)
        assert p.shape == (2, 2)
        assert np.allclose(p, [[0, d], [d, 0]])

@norm
def test_sparse_pairwise(normalize):
    leven = Levenshtein(normalize=normalize)
    for (x, y), d in EXAMPLES.items():
        if normalize:
            d /= max(len(x), len(y))
        p = leven.sparse_pairwise((x, y))
        assert isinstance(p, sp.csr_matrix)
        assert p.shape == (2, 2)
        assert p.nnz == 1
        assert np.all(p.data == d)
