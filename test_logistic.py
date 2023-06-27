import pytest
from logistic import logistic_map, iterate_f
from numpy.testing import assert_allclose
from math import isclose

@pytest.mark.parametrize(
	"x, r, expected", 
	[(0.1,2.2, 0.198),(0.2,3.4, 0.544),(0.75,1.7,0.31875)])


def test_logistic(x,r, expected):
	results = logistic_map(x,r)
	assert isclose(results, expected)
