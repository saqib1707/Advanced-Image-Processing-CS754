import numpy as np
from scipy.fftpack import dct, idct
import math

N=8
a = np.zeros((N, N))
for i in range(N):
	for j in range(N):
		if i == 0:
			a[i][j] = 1.0/math.sqrt(N)
		else:
			a[i][j] = (math.sqrt(2.0/N))*math.cos(((2*j+1)*i*math.pi)/(2.0*N))

sig2 = np.array([4,2,7,1,9,10,12,3])
mydctcoff = np.matmul(a, sig2)

dctcoff = dct(sig2,norm='ortho')
print(mydctcoff)
print(dctcoff)