"""
Created 06. June 2024 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import glob
import fabio
import pyFAI
import pygix
import scipy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pyFAI import azimuthalIntegrator as pa
from pygix import lattice as pl
from pygix import plotting as pp

def take(img,headerkey,indices):
	return np.fromstring(img.header[headerkey],sep=' ')[indices]

for f in glob.glob('*.img'):
	img=fabio.open(f);filename=os.path.splitext(f)[0].replace('_image','')
	detdist=take(img,'PXD_GONIO_VALUES',-1);beamcenterX,beamcenterY,pxsizeX,pxsizeY=take(img,'PXD_SPATIAL_DISTORTION_INFO',[0,1,2,3]);wavelength=take(img,'SOURCE_WAVELENGTH',-1);omega,chi,phi=take(img,'CRYSTAL_GONIO_VALUES',[0,1,2])
	ai=pa.AzimuthalIntegrator(
	dist=detdist/1e3,poni1=beamcenterY*pxsizeY/1e3,poni2=beamcenterX*pxsizeX/1e3,pixel1=pxsizeY/1e3,pixel2=pxsizeX/1e3,wavelength=wavelength/1e10)
	pg=pygix.Transform(
	dist=detdist/1e3,poni1=beamcenterY*pxsizeY/1e3,poni2=beamcenterX*pxsizeX/1e3,pixel1=pxsizeY/1e3,pixel2=pxsizeX/1e3,wavelength=wavelength/1e10,incident_angle=0,sample_orientation=1,tilt_angle=0,useqx=True)
	npts=max(img.shape)//2

	#AI
	plt.close('all')
	plt.figure(figsize=(7.5/2.54,5.3/2.54))
	mpl.rc('text',usetex=True);mpl.rc('text.latex',preamble=r'\usepackage[helvet]{sfmath}')
	yobs,q,chi=ai.integrate2d(img.data,npt_rad=npts,npt_azim=npts)
	plt.pcolormesh(q,chi,yobs,cmap='coolwarm')
	plt.xlabel(r'$q/\rm{nm}^{-1}$')
	plt.ylabel(r'$\chi/^\circ$')
	plt.tick_params(axis='both',pad=2,labelsize=8)
	plt.tight_layout(pad=0.1)
	plt.savefig(filename+'_ai2d.png',dpi=300)

	p1_ranges=[(-10,10),(80,100)]
	for p1_range in p1_ranges:
		plt.close('all')
		plt.figure(figsize=(7.5/2.54,5.3/2.54))
		mpl.rc('text',usetex=True);mpl.rc('text.latex',preamble=r'\usepackage[helvet]{sfmath}')
		q,yobs=ai.integrate1d(img.data,npt=npts,azimuth_range=p1_range)
		plt.plot(q,yobs/max(yobs))
		plt.figtext(0.98,0.98,r'$\chi='+str((p1_range[0],p1_range[1]))+'$',ha='right',va='top',fontsize=6)
		plt.xlabel(r'$q/\rm{nm}^{-1}$')
		plt.ylabel(r'$I/1$')
		plt.tick_params(axis='both',pad=2,labelsize=8)
		plt.tight_layout(pad=0.1)
		plt.savefig(filename+'_'+str(p1_range)+'_aiIq.png',dpi=300)

	#PG
	plt.close('all')
	plt.figure(figsize=(7.5/2.54,5.3/2.54))
	mpl.rc('text',usetex=True);mpl.rc('text.latex',preamble=r'\usepackage[helvet]{sfmath}')
	yobs,q,chi=pg.transform_polar(img.data,npt=(npts,npts),method='bbox')
	qg,chig=np.meshgrid(q,chi);yobs[np.where(abs(chig)==np.min(abs(chig)))]=0
	plt.pcolormesh(q,chi,yobs,cmap='coolwarm')
	plt.xlabel(r'$q/\rm{nm}^{-1}$')
	plt.ylabel(r'$\chi/^\circ$')
	plt.tick_params(axis='both',pad=2,labelsize=8)
	plt.tight_layout(pad=0.1)
	plt.savefig(filename+'_pg2d.png',dpi=300)

	plt.close('all')
	plt.figure(figsize=(7.5/2.54,5.3/2.54))
	mpl.rc('text',usetex=True);mpl.rc('text.latex',preamble=r'\usepackage[helvet]{sfmath}')
	yobs,qxy,qz=pg.transform_reciprocal(img.data,npt=(npts,npts),method='bbox')
	qxyg,qzg=np.meshgrid(qxy,qz);yobs[np.where(abs(qxyg)==np.min(abs(qxyg)))]=0
	plt.pcolormesh(qxy,qz,yobs,cmap='coolwarm')
	plt.axis('equal')
	plt.xlabel(r'$q_{xy}/\rm{nm}^{-1}$')
	plt.ylabel(r'$q_z/\rm{nm}^{-1}$')
	plt.tick_params(axis='both',pad=2,labelsize=8)
	plt.tight_layout(pad=0.1)
	plt.savefig(filename+'_pgrs.png',dpi=300)

	qz_props=[(0.1,-0.1),(-0.1,0.1)]
	for qz_prop in qz_props:
		plt.close('all')
		plt.figure(figsize=(7.5/2.54,5.3/2.54))
		mpl.rc('text',usetex=True);mpl.rc('text.latex',preamble=r'\usepackage[helvet]{sfmath}')
		qg=(qxyg**2+qzg**2)**0.5
		args=np.where((qzg/qg>=qz_prop[0])&(qzg/qg<=qz_prop[1]))
		yint,q=np.histogram(qg[args].flatten(),weights=yobs[args].flatten(),bins=npts//2)
		plt.plot(q[1:],yint/np.nan_to_num(np.max(yint)))
		plt.figtext(0.98,0.98,r'$q_z/q='+str((qz_prop[0],qz_prop[1]))+'$',ha='right',va='top',fontsize=6)
		plt.xlabel(r'$q/\rm{nm}^{-1}$')
		plt.ylabel(r'$I/1$')
		plt.tick_params(axis='both',pad=2,labelsize=8)
		plt.tight_layout(pad=0.1)
		plt.savefig(filename+'_'+str(qz_prop)+'_pgIq.png',dpi=300)
