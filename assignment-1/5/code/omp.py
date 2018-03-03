import numpy as np 

y = np.array([2.7,0.1,4.5])
final_phi = np.array([[-0.8,0.3,1,0.4],[-0.2,0.4,-0.3,-0.4],[0.2,1,-0.1,0.8]])
xrec = np.zeros((final_phi.shape[1],1))
y = np.reshape(y, (y.shape[0],1))
r = y
epsilon = pow(10,-10)
tset = []
itr=0

norm = np.linalg.norm(final_phi,ord=2,axis=0)
norm_phi = np.divide(final_phi, norm)

while(np.linalg.norm(r)**2 > epsilon):
	print("Iteration:",itr)
	itr+=1
	dot = np.matmul(np.transpose(r), norm_phi)
	print("Dot:",dot)
	index=np.argmax(np.abs(dot))
	tset = np.append(tset, index)
	tset = np.sort(tset)
	tset = tset.astype(int)
	Anew = final_phi[:,tset]
	print("Anew:",Anew)
	Anewplus = np.linalg.pinv(Anew)
	lambd = np.matmul(Anewplus, y)
	print("Lambda:", lambd)
	xrec[tset] = lambd
	print("xrec:",xrec)
	r = y - np.matmul(Anew, lambd)
	print("Residue:",r)

print("Total Iterations Taken:",itr)
print("Reconstructed X")
print(xrec)