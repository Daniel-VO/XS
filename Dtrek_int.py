import os
import glob
import fabio
import pygix
import numpy as np
import pygix.plotting as pp

def take(img,headerkey,indices):
	return np.fromstring(img.header[headerkey],sep=' ')[indices]

for f in glob.glob('*.img'):
	filename=os.path.splitext(f)[0].replace('_image','')
	img=fabio.open(f)

	detdist=take(img,'PXD_GONIO_VALUES',-1);beamcenterX,beamcenterY,pxsizeX,pxsizeY=take(img,'PXD_SPATIAL_DISTORTION_INFO',[0,1,2,3]);wavelength=take(img,'SOURCE_WAVELENGTH',-1);omega,chi,phi=take(img,'CRYSTAL_GONIO_VALUES',[0,1,2])

	pg=pygix.Transform(dist=detdist/1e3,poni1=beamcenterY*pxsizeY/1e3,poni2=beamcenterX*pxsizeX/1e3,pixel1=pxsizeY/1e3,pixel2=pxsizeX/1e3,wavelength=wavelength/1e10,incident_angle=0)

	i,qxy,qz=pg.transform_reciprocal(img.data)

	pp.implot(i,qxy,qz,xlim=(0,None),mode='rsm')
