"""
Created 10. June 2024 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import sys
import glob
import fabio
import pyFAI
import scipy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pyFAI import azimuthalIntegrator as pa

if len(sys.argv)>1:
	geom=sys.argv[1]
else:
	geom=input('Geometrie [Faser/GISAXS/ ]: ')

def label(string):
	return r'$'+string.replace('^-1','^{-1}').replace('_deg','/^\circ').replace('A',r'\rm{A}').replace('nm',r'\rm{nm}').replace('_','}/').replace('q','q_{').replace('th',r'\theta')+'$'

def take(img,headerkey,indices):
	return np.fromstring(img.header[headerkey],sep=' ')[indices]

for f in glob.glob('*.img'):
	img=fabio.open(f);filename=os.path.splitext(f)[0].replace('_image','')
	detdist=take(img,'PXD_GONIO_VALUES',-1);beamcenterX,beamcenterY,pxsizeX,pxsizeY=take(img,'PXD_SPATIAL_DISTORTION_INFO',[0,1,2,3]);wavelength=take(img,'SOURCE_WAVELENGTH',-1);omega,chi,phi=take(img,'CRYSTAL_GONIO_VALUES',[0,1,2])
	ai=pa.AzimuthalIntegrator(dist=detdist/1e3,poni1=beamcenterY*pxsizeY/1e3,poni2=beamcenterX*pxsizeX/1e3,pixel1=pxsizeY/1e3,pixel2=pxsizeX/1e3,wavelength=wavelength/1e10)

	if geom=='Faser':
		ta=0;so=1
		unit_qip=pyFAI.units.get_unit_fiber('qip_A^-1',incident_angle=np.radians(omega),tilt_angle=np.radians(ta),sample_orientation=so)
		unit_qoop=pyFAI.units.get_unit_fiber('qoop_A^-1',incident_angle=np.radians(omega),tilt_angle=np.radians(ta),sample_orientation=so)
		units=(unit_qip,unit_qoop);method='no';npts=(img.shape[1]//2,img.shape[0]//2)
	else:
		units='q_A^-1';method='splitpixel';npts=(img.shape[1]//2,360)

	#2D
	plt.close('all')
	plt.figure(figsize=(7.5/2.54,5.3/2.54))
	mpl.rc('text',usetex=True);mpl.rc('text.latex',preamble=r'\usepackage[helvet]{sfmath}')

	yobs,x,y=ai.integrate2d(img.data,npts[0],npts[1],unit=units,method=method)

	if geom=='Faser':
		xg,yg=np.meshgrid(x,y);args=np.where(abs(xg)==np.min(abs(xg)))
		yobs[args]=np.median([yobs[args[0],args[1]-1],yobs[args[0],args[1]+1]],axis=0)
		xlabel=label(str(units[0]));ylabel=label(str(units[1]))
	else:
		xlabel=label(units);ylabel=r'$\chi/^\circ$'

	plt.pcolormesh(x,y,yobs,cmap='coolwarm')
	plt.xlabel(xlabel);plt.ylabel(ylabel)
	plt.tick_params(axis='both',pad=2,labelsize=8)
	plt.tight_layout(pad=0.1)
	plt.savefig(filename+'.png',dpi=300)

	#1D
	plt.close('all')
	plt.figure(figsize=(7.5/2.54,5.3/2.54))
	mpl.rc('text',usetex=True);mpl.rc('text.latex',preamble=r'\usepackage[helvet]{sfmath}')

	if geom=='Faser':
		rang=None
		x,yobs=ai.integrate_fiber(img.data,npt_output=npts[0],output_unit=units[0],integrated_unit=units[1],integrated_unit_range=rang,method=method,filename=filename+'_'+str(rang)+'.xy')
		yobs[np.where(abs(x)==min(abs(x)))]=0
		xlabel=label(str(units[0]));ylabel=r'$I/1$'
	else:
		rang=None
		x,yobs=ai.integrate1d(img.data,npt=npts[0],azimuth_range=rang,unit=units,method=method,filename=filename+'_'+str(rang)+'.xy')
		xlabel=label(units);ylabel=r'$I/1$'

	plt.plot(x,yobs,'k',linewidth=0.5)

	plt.xlabel(xlabel);plt.ylabel(ylabel)
	plt.tick_params(axis='both',pad=2,labelsize=8)
	plt.tight_layout(pad=0.1)
	plt.savefig(filename+'_'+str(rang)+'.png',dpi=300)


