"""
Created 08. April 2024 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import glob
import scipy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import fabio
from Dtrek_int import take,center_pad

filenamepatterns=['*']

for fnp in filenamepatterns:
	stack=[]
	for f in glob.glob(fnp+'[!*].img'):
		img=fabio.open(f)
		# ~ print(img.header,file=open('header','w'));a=b
		detdist=take(img,'PXD_GONIO_VALUES',-1);detsizeX,detsizeY=take(img,'PXD_DETECTOR_SIZE',[0,1]);beamcenterX,beamcenterY,pxsizeX,pxsizeY=take(img,'PXD_SPATIAL_DISTORTION_INFO',[0,1,2,3]);wavelength=take(img,'SOURCE_WAVELENGTH',-1);omega,chi,phi=take(img,'CRYSTAL_GONIO_VALUES',[0,1,2])

		if 'SCAN_DET_ROTATION' in img.header:
			ttmin,ttmax=take(img,'SCAN_DET_ROTATION',[0,1])
			beamcenterX=img.shape[1]*ttmin/(ttmin-ttmax)

			img.data=center_pad(img,beamcenterX,beamcenterY)

			fs=(img.shape[0]**2+img.shape[1]**2)**0.5
			img.data=scipy.ndimage.rotate(img.data,chi)
			img.data=img.data[:img.shape[0]//2*2,:img.shape[1]//2*2]
			pady=int((fs-img.shape[0])/2);padx=int((fs-img.shape[1])/2)
			img.data=np.pad(img.data,((pady,pady),(padx,padx)))

		stack.append(img.data)

	fnp=fnp.replace('*','alle')
	stack=np.max(stack,axis=0)
	plt.imsave(fnp+'_stack.png',stack,cmap='coolwarm')
	fabio.dtrekimage.DtrekImage(data=stack.astype(np.float32),header=img.header).write(fnp+'.img')
