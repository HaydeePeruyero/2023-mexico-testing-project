import pytest
from logistic import f, iterate_f, random_generator
from numpy.testing import assert_allclose
from math import isclose
import numpy as np


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
	



@pytest.fixture
def random_state():
	SEED = np.random.randint(0, 2**31)
	print(f'Using sedd {SEED}')
	rs = np.random.RandomState(SEED)
	return rs


def test_converge(random_state):
	#SEED = 41
	#x0 = random_generator(SEED)
	x0 = random_state.rand()
	result = iterate_f(100, x0, 1.5)
	assert isclose(result[-1], 1/3)
	
	
def test_con():
	seed = 1213
	random_state = np.random.RandomState(seed)
	x0 = random_state.rand()
	xs = iterate_f(100,x0,1.5)
	expected=1/3
	assert isclose(xs[-1],expected) 
	

	

