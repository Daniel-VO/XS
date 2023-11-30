"""
Created 30. November 2023 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import fabio,pyFAI
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from pyFAI.engines.CSR_engine import CSRIntegrator
from pyFAI.opencl.peak_finder import OCL_PeakFinder

def take(headerkey,indices):
	return np.fromstring(img.header[headerkey],sep=' ')[indices]

os.system('mkdir norefs')
os.system('rm *.jpg')
reflections=[]
unit='q_A^-1'

for f in glob.glob('*.img'):
	filename=os.path.splitext(f)[0];img=fabio.open(f)
	# ~ print(img.header,file=open('header','w'));a=b

	plt.close('all')
	plt.imshow(img.data,cmap='coolwarm')
	plt.gca().invert_yaxis()

	if np.array(np.where(img.data>np.max(img.data)/2)).flatten().shape[0]/img.data.flatten().shape[0]<2e-3:
		twotheta,detdist=take('PXD_GONIO_VALUES',[1,-1]);detsizeX,detsizeY=take('PXD_DETECTOR_SIZE',[0,1]);beamcenterX,beamcenterY,pxsizeX,pxsizeY=take('PXD_SPATIAL_DISTORTION_INFO',[0,1,2,3]);wavelength=take('SOURCE_WAVELENGTH',-1);omega,chi,phi=take('CRYSTAL_GONIO_VALUES',[0,1,2])
		ai=AzimuthalIntegrator(dist=detdist/1e3,poni1=beamcenterY*pxsizeY/1e3,poni2=beamcenterX*pxsizeX/1e3,pixel1=pxsizeY/1e3,pixel2=pxsizeX/1e3,wavelength=wavelength/1e10,rot1=0,rot2=np.radians(twotheta),rot3=0)

		sigma_clip=ai.sigma_clip_ng(img.data,error_model='azimuthal',unit=unit)
		pf=OCL_PeakFinder(ai.engines[sigma_clip.method].engine.lut,image_size=np.prod(img.shape),bin_centers=sigma_clip.radial,radius=ai.array_from_unit(unit=sigma_clip.unit),unit=unit)
		peaks=pf.peakfinder8(img.data,error_model='azimuthal',connected=50,patch_size=50)

		twotheta0,omega0,chi0,phi0=np.radians([twotheta,omega,chi,phi])
		for p in peaks:
			yobs,sig,pos0,pos1=list(p)[1:]
			sig=np.arctan(sig*pxsizeY/detdist)
			dangY=np.arctan((pos0-beamcenterY)*pxsizeY/detdist);dangX=np.arctan((pos1-beamcenterX)*pxsizeX/detdist)

			twotheta=np.arccos(np.cos(twotheta0+dangY)*np.cos(dangX))
			omega=-omega0+twotheta/2
			chi=chi0+dangX/2*np.cos(omega)
			phi=phi0+dangX/2*np.sin(omega)

			q=4*np.pi*np.sin(twotheta/2)/wavelength
			qx=q*((np.sin(chi)*np.sin(phi))**2+(np.sin(omega)*np.cos(phi))**2)**0.5\
				*np.sign(np.sin(chi)*np.sin(phi)+np.sin(omega)*np.cos(phi))
			qy=q*((np.sin(chi)*np.cos(phi))**2+(np.sin(omega)*np.sin(phi))**2)**0.5\
				*np.sign(np.sin(chi)*np.cos(phi)+np.sin(omega)*np.sin(phi))
			qz=q*np.cos(chi)*np.cos(omega)
			# ~ print(q,np.linalg.norm([qx,qy,qz]),1-q/np.linalg.norm([qx,qy,qz]))

			reflections.append([filename,q,qx,qy,qz,yobs,sig])

		plt.plot(peaks['pos1'],peaks['pos0'],'.w')
		plt.xlabel('X');plt.ylabel('Y')
		plt.tight_layout(pad=0.1)
		plt.savefig(filename+'.jpg')
	else:
		os.system('mv '+f+' norefs/')
		plt.tight_layout(pad=0.1)
		plt.savefig('norefs/'+filename+'.jpg')

np.save('reflections.npy',reflections)
os.system('python3 refindex_Dtrek_img.py')
