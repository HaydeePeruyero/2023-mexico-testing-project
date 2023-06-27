def logistic_map(x,r):
	#fx = r * x * (1 - x)
	
	return r * x * (1 - x)

def iterate_f(it, xi, r):
	x = xi
	xs = []
	for _ in range(it):
		x = logistic_map(x,r)
		xs.append(x)
	
	return np.array(xs)
