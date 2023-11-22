"""
Created 21. November 2023 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import numpy as np
import os
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import fabio,pyFAI
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from pyFAI.engines.CSR_engine import CSRIntegrator
from pyFAI.opencl.peak_finder import OCL_PeakFinder

def take(headerkey,indices):
	return np.fromstring(img.header[headerkey],sep=' ')[indices]

reflections=[]
unit='q_A^-1'

for f in glob.glob('*.img'):
	filename=os.path.splitext(f)[0];img=fabio.open(f)
	# ~ print(img.header,file=open('header','w'));a=b
	if np.array(np.where(img.data>np.max(img.data)/2)).flatten().shape[0]/img.data.flatten().shape[0]<1e-3:
		twotheta,detdist=take('PXD_GONIO_VALUES',[1,-1]);detsizeX,detsizeY=take('PXD_DETECTOR_SIZE',[0,1]);beamcenterX,beamcenterY,pxsizeX,pxsizeY=take('PXD_SPATIAL_DISTORTION_INFO',[0,1,2,3]);wavelength=take('SOURCE_WAVELENGTH',-1);omega,chi,phi=take('CRYSTAL_GONIO_VALUES',[0,1,2])
		ai=AzimuthalIntegrator(dist=detdist/1e3,poni1=beamcenterY*pxsizeY/1e3,poni2=beamcenterX*pxsizeX/1e3,pixel1=pxsizeY/1e3,pixel2=pxsizeX/1e3,wavelength=wavelength/1e10,rot1=0,rot2=np.radians(twotheta),rot3=0)

		sigma_clip=ai.sigma_clip_ng(img.data,error_model='azimuthal',unit=unit)
		pf=OCL_PeakFinder(ai.engines[sigma_clip.method].engine.lut,image_size=np.prod(img.shape),bin_centers=sigma_clip.radial,radius=ai.array_from_unit(unit=sigma_clip.unit),unit=unit)
		peaks=pf.peakfinder8(img.data,error_model='azimuthal',connected=50,patch_size=50)

		plt.close('all')
		plt.imshow(img.data,cmap='coolwarm')
		plt.plot(peaks['pos1'],peaks['pos0'],'.w')
		plt.savefig(filename+'.png',dpi=300)

		twotheta0,chi0,phi0=np.radians([twotheta,chi,phi])
		for p in peaks:
			yobs,sig,pos0,pos1=list(p)[1:]
			dangY=np.arctan((pos0-beamcenterY)*pxsizeY/detdist);dangX=np.arctan((pos1-beamcenterX)*pxsizeX/detdist)

			twotheta=np.arctan((np.tan(twotheta0+dangY)**2+np.tan(dangX)**2)**0.5)
			omega=(twotheta0-twotheta)/2
			chi=chi0+dangX/2
			phi=phi0

			q=4*np.pi*np.sin(twotheta/2)/wavelength
			qx=-np.sin(omega)*np.cos(phi)-np.sin(chi)*np.sin(phi)
			qy=-np.sin(omega)*np.sin(phi)-np.sin(chi)*np.cos(phi)
			qz= np.cos(omega)*np.cos(chi)
			qx,qy,qz=q*np.array([qx,qy,qz])/np.linalg.norm([qx,qy,qz])

			reflections.append([filename,q,qx,qy,qz,yobs,sig])

np.save('reflections.npy',reflections)

