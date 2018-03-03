import cv2
import numpy as np

cap = cv2.VideoCapture('cars.avi')
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
	# if cv2.waitKey(1) & 0xFF == ord('q'):
	# 	break
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()

height,width = images[0].shape   # always returns (rows, columns, channel)
coded_pattern = np.random.random_integers(low=0, high=1, size=(height, width, T))

Ixy=np.zeros((height,width))

for i in range(3):
	Ixy = np.add(Ixy, np.multiply(coded_pattern[:,:,i], images[i]))

coded_snapshot = Ixy/3.0     # coded snapshot
mu, sigma = 0.0, 2.0
noise = np.random.normal(mu, sigma, size=(height, width))
#Ixy = np.add(Ixy, noise)              # coded snapshot which is my measurement

cv2.imshow('coded snapshot', coded_snapshot/255.0)
cv2.waitKey(0)
cv2.destroyAllWindows()
finalImages = np.zeros(coded_pattern.shape)

# Reconstruction Algorithm
patch_size=8

for i in range(15):
	for j in range(30):
		y_patch = Ixy[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
		binary_patch = coded_pattern[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size, :]
		s1 = np.zeros((patch_size*patch_size, patch_size*patch_size))
		s2 = np.zeros((patch_size*patch_size, patch_size*patch_size))
		s3 = np.zeros((patch_size*patch_size, patch_size*patch_size))
		np.fill_diagonal(s1, binary_patch[:,:,0])
		np.fill_diagonal(s2, binary_patch[:,:,1])
		np.fill_diagonal(s3, binary_patch[:,:,2])
		
		final_phi = np.hstack((np.hstack((s1,s2)),s3))
		y = np.reshape(y_patch, (patch_size*patch_size))
		xrec = np.zeros((final_phi.shape[1],1))

		r = y
		epsilon = 0
		tset = []
		itr=0

		dot = np.matmul(np.transpose(r), final_phi)
		index=np.argmax(np.abs(dot))
		tset = np.append(tset,index)
		r = y - dot[index]*final_phi[:,index]
		Anew = final_phi[:,index]
		xrec[index,0] = dot[index]
		Anew = np.reshape(Anew, (Anew.shape[0],1))

		while(np.linalg.norm(r)**2 > epsilon):
			itr+=1
			temp = np.matmul(np.transpose(r), final_phi)
			index=np.argmax(np.abs(temp))
			tset = np.append(tset, index)
			tset = np.sort(tset)
			tset=tset.astype(int)
			Anew = final_phi[:,tset]
			Anewplus = np.linalg.pinv(Anew)
			lambd = np.matmul(Anewplus, y)
			xrec[tset,0] = lambd
			r = y - np.matmul(Anew, lambd)
		shape = 64
		finalImages[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size, 0] = np.reshape(xrec[0:shape,0], (patch_size, patch_size))
		finalImages[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size, 1] = np.reshape(xrec[shape:2*shape,0],(patch_size, patch_size))
		finalImages[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size, 2] = np.reshape(xrec[2*shape:3*shape,0],(patch_size, patch_size))

		print(i, j)

print(np.max(finalImages))
cv2.imshow('original images[0]', images[0])
cv2.imshow('original images[1]', images[1])
cv2.imshow('original images[2]', images[2])
cv2.imshow('coded snapshot', Ixy/255.0)
cv2.imshow('finalImage[0]', finalImages[:,:,0]/(255.0*3.0))
cv2.imshow('finalImage[1]', finalImages[:,:,1]/(255.0*3.0))
cv2.imshow('finalImage[2]', finalImages[:,:,2]/(255.0*3.0))
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Done")