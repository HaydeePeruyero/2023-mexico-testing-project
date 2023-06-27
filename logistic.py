import numpy as np

def f(x,r):
	#fx = r * x * (1 - x)
	
	return r * x * (1 - x)

def iterate_f(it, xi, r):
	x = xi
	xs = []
	for _ in range(it):
		x = f(x,r)
		xs.append(x)
	
	return np.array(xs)
	
def random_generator(SEED):
	random_state = np.random.RandomState(SEED)
	return random_state.rand()
