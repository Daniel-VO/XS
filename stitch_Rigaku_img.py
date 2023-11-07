"""
Created 07. November 2023 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import glob
import numpy as np
import scipy
import fabio,imageio
import matplotlib as mpl
import matplotlib.pyplot as plt

def take(headerkey,indices):
	return np.fromstring(img.header[headerkey],sep=' ')[indices]

stack=[]
for f in glob.glob('*.img'):
	img=fabio.open(f)
	detdist=take('PXD_GONIO_VALUES',-1);detsizeX,detsizeY=take('PXD_DETECTOR_SIZE',[0,1]);beamcenterX,beamcenterY,pxsizeX,pxsizeY=take('PXD_SPATIAL_DISTORTION_INFO',[0,1,2,3]);wavelength=take('SOURCE_WAVELENGTH',-1);omega,chi,phi=take('CRYSTAL_GONIO_VALUES',[0,1,2])
	ttmin,ttmax=take('SCAN_DET_ROTATION',[0,1])
	img.data=np.pad(img.data,((0,0),(int(img.shape[1]+2*ttmin*img.shape[1]/(ttmax-ttmin)),0)))
	img.data=scipy.ndimage.rotate(img.data,-90-chi)
	fs=2300
	img.data=np.pad(img.data,(((fs-img.shape[0])//2,(fs-img.shape[0])//2),((fs-img.shape[1])//2,(fs-img.shape[1])//2)))
	stack.append(img.data)

stack=np.max(stack,axis=0)
plt.imsave('stack.png',stack,cmap='coolwarm')
