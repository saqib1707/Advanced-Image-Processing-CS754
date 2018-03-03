import cv2
import numpy as np
import math

cap = cv2.VideoCapture('../data/cars.avi')
count=0
T=3
images = []
while(cap.isOpened()):
	count+=1
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = gray[168:288, 112:352]
	images.append(gray)
	if(count >= T):
		break

cap.release()
cv2.destroyAllWindows()

height,width = images[0].shape   # always returns (rows, columns, channel)
coded_pattern = np.random.random_integers(low=0, high=1, size=(height, width, T))
Ixy=np.zeros((height,width))

for frame in range(T):
	images[frame] = cv2.normalize(images[frame].astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
	Ixy = np.add(Ixy, np.multiply(coded_pattern[:,:,frame], images[frame]))

Ixy = Ixy/3.0     # coded snapshot
mu, sigma = 0.0, 2.0
noise = np.random.normal(mu, sigma, size=(height, width))
#Ixy = np.add(Ixy, noise)              # coded snapshot which is my measurement

# cv2.imshow('coded snapshot', Ixy)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

finalImages = np.zeros(coded_pattern.shape)

# Reconstruction Algorithm
N=8
a = np.zeros((N, N))
for i in range(N):
	for j in range(N):
		if i == 0:
			a[i][j] = 1.0/math.sqrt(N)
		else:
			a[i][j] = (math.sqrt(2.0/N))*math.cos((math.pi*(2*j+1)*i*math.pi)/(2.0*N*180.0))

dct = np.kron(np.transpose(a),a)
# cv2.imshow('dct', dct)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
patch_size=8

for i in range(15):
	for j in range(30):
		y_patch = Ixy[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
		binary_patch = coded_pattern[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size, :]

		for frame in range(T):
			si = np.zeros((patch_size*patch_size, patch_size*patch_size))
			np.fill_diagonal(si, binary_patch[:,:,frame])
			si = np.matmul(si, dct)
			if frame == 0:
				final_phi = si
			else:
				final_phi = np.hstack((final_phi, si))

		y = np.reshape(y_patch, (patch_size*patch_size,1))
		xrec = np.zeros((final_phi.shape[1],1))

		r = y
		epsilon = pow(10,-10)
		tset = []
		itr=0

		norm = np.linalg.norm(final_phi,ord=2,axis=0)
		norm_phi = np.divide(final_phi, norm)

		while(np.linalg.norm(r)**2 > epsilon):
		#while(itr < 100):
			#print(np.linalg.norm(r)**2)
			itr+=1
			dot = np.matmul(np.transpose(r), norm_phi)
			index=np.argmax(np.abs(dot))
			tset = np.append(tset, index)
			tset = np.sort(tset)
			tset = tset.astype(int)
			Anew = final_phi[:,tset]
			Anewplus = np.linalg.pinv(Anew)
			lambd = np.matmul(Anewplus, y)
			xrec[tset] = lambd
			r = y - np.matmul(Anew, lambd)

		shape = patch_size*patch_size

		for frame in range(T):
			xrec[frame*64:(frame+1)*64,0] = np.matmul(dct, xrec[frame*64:(frame+1)*64,0])

		for frame in range(T):
			finalImages[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size, frame] = np.reshape(xrec[frame*shape:(frame+1)*shape,0], (patch_size, patch_size))

		print(i, j)

print(np.max(finalImages))
for frame in range(T):
	cv2.imshow('original images[{}]'.format(frame), images[frame])
	cv2.imshow('finalImage[{}]'.format(frame), finalImages[:,:,frame])
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Done")


recError = 0
# for finding Reconstruction error
for frame in range(T):
	recError += np.sum(np.square((images[frame] - finalImages[:,:,frame])))
print('Reconstruction Error for T frames:', recError)