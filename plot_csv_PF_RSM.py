import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

for f in glob.glob('*.csv'):
	filename=os.path.splitext(f)[0]
	print(filename)

	plt.close('all')

	if 'RSM' in open(f).readlines()[0]:
		deltaphi_deg,ttchi_deg,yobs=np.genfromtxt(f,delimiter=',',unpack=True,skip_header=2)
		Y,X=np.meshgrid(np.unique(2*np.sin(np.radians(ttchi_deg/2))/1.5406),np.unique(deltaphi_deg))
	else:
		psi_deg,phi_deg,yobs=np.genfromtxt(f,delimiter=',',unpack=True)
		X,Y=np.meshgrid(np.unique(np.radians(phi_deg)),np.unique(psi_deg))
		plt.subplot(projection='polar')

	plt.contourf(X,Y,np.log(np.reshape(yobs,X.shape)),cmap='coolwarm')

	plt.savefig(filename+'.pdf')
	plt.savefig(filename+'.png')
