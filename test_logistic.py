import pytest
from logistic import f, iterate_f, random_generator
from numpy.testing import assert_allclose
from math import isclose

@pytest.mark.xfail
@pytest.mark.parametrize("a", [1,2,3])
@pytest.mark.parametrize("b", [2,5,3])
def test_sums(a,b):
	assert a + b < a

@pytest.mark.parametrize(
	"x, r, expected", 
	[(0.1,2.2, 0.198),(0.2,3.4, 0.544),(0.75,1.7,0.31875)])
def test_logistic(x,r, expected):
	results = f(x,r)
	assert isclose(results, expected)
	

def test_converge():
	SEED = 41
	x0 = random_generator(SEED)
	result = iterate_f(100, x0, 1.5)
	assert isclose(result[99], 1/3)
	
	

