"""
Created 27. March 2024 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import fabio

def take(headerkey,indices):
	return np.fromstring(img.header[headerkey],sep=' ')[indices]

filenamepatterns=['*']

for fnp in filenamepatterns:
	stack=[]
	for f in glob.glob(fnp+'.img'):
		img=fabio.open(f)
		# ~ print(img.header,file=open('header','w'));a=b
		detdist=take('PXD_GONIO_VALUES',-1);detsizeX,detsizeY=take('PXD_DETECTOR_SIZE',[0,1]);beamcenterX,beamcenterY,pxsizeX,pxsizeY=take('PXD_SPATIAL_DISTORTION_INFO',[0,1,2,3]);wavelength=take('SOURCE_WAVELENGTH',-1);omega,chi,phi=take('CRYSTAL_GONIO_VALUES',[0,1,2])

		ttmin,ttmax=take('SCAN_DET_ROTATION',[0,1])
		img.data=np.pad(img.data,((0,0),(int(img.shape[1]+2*ttmin*img.shape[1]/(ttmax-ttmin)),0)))
		fs=(img.shape[0]**2+img.shape[1]**2)**0.5
		img.data=scipy.ndimage.rotate(img.data,-90-chi)
		img.data=img.data[:img.shape[0]//2*2,:img.shape[1]//2*2]
		padx=int((fs-img.shape[0])/2);pady=int((fs-img.shape[1])/2)
		img.data=np.pad(img.data,((padx,padx),(pady,pady)))
		stack.append(img.data)

	stack=np.max(stack,axis=0)/np.max(stack)
	plt.imsave(fnp+'.png',stack,cmap='coolwarm')
	fabio.dtrekimage.DtrekImage(data=stack,header=img.header).write(fnp+'.img')
