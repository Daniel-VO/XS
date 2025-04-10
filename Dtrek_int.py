"""
Created 07. April 2025 by Daniel Van Opdenbosch, Technical University of Munich

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
import pybaselines as pb;bl=pb.api.Baseline()
import pyFAI.integrator.azimuthal as pa
from pyFAI.units import get_unit_fiber

if len(sys.argv)>1:
	geom=sys.argv[1]
else:
	geom=''

def label(string):
	return r'$'+str(string).replace('^-1','^{-1}').replace('_deg','/^\circ').replace('A',r'\rm{\AA}').replace('nm',r'\rm{nm}').replace('_','}}/').replace('th',r'\theta').replace('q','q_{\mathrm{').replace('chi',r'\chi')+'$'

def take(img,headerkey,indices):
	return np.fromstring(img.header[headerkey],sep=' ')[indices]

SAXSmin=1e-1																	####

for f in glob.glob('*.img'):
	img=fabio.open(f);filename=os.path.splitext(f)[0].replace('_image','');print(filename)
	detdist=take(img,'PXD_GONIO_VALUES',-1);beamcenterX,beamcenterY,pxsizeX,pxsizeY=take(img,'PXD_SPATIAL_DISTORTION_INFO',[0,1,2,3]);wavelength=take(img,'SOURCE_WAVELENGTH',-1);omega,chi,phi=take(img,'CRYSTAL_GONIO_VALUES',[0,1,2])
	ai=pa.AzimuthalIntegrator(dist=detdist/1e3,poni1=beamcenterY*pxsizeY/1e3,poni2=beamcenterX*pxsizeX/1e3,pixel1=pxsizeY/1e3,pixel2=pxsizeX/1e3,wavelength=wavelength/1e10)

	if geom=='Faser':
		ta=0;so=int(sys.argv[2])
		unit_qip=get_unit_fiber('qip_A^-1',incident_angle=np.radians(omega),tilt_angle=np.radians(ta),sample_orientation=so)
		unit_qoop=get_unit_fiber('qoop_A^-1',incident_angle=np.radians(omega),tilt_angle=np.radians(ta),sample_orientation=so)
		units=(unit_qip,unit_qoop);method='no';npts=(img.shape[1]//2,img.shape[0]//2)
		if so==2 or so==4:
			npts=npts[::-1]
	else:
		units='2th_deg';method='splitpixel';npts=(img.shape[1]//2,360)

	unit='2th_deg'
	if len(glob.glob('*_BG_image.img'))>0:
		bg=glob.glob('*_BG_image.img')[0];print('Untergrund: '+bg)
		img.data=img.data.astype('float')-fabio.open(bg).data.astype('float')

	plt.close('all')
	mfilt1d=ai.medfilt1d(img.data,npt_rad=npts[0],npt_azim=npts[1],unit=unit,method=method,percentile=5)
	baseline=bl.irsqr(mfilt1d.intensity,diff_order=2)[0]
	isotropic=ai.calcfrom1d(mfilt1d.radial,baseline,shape=img.shape,dim1_unit=mfilt1d.unit)
	int1d=ai.integrate1d(img.data,npt=npts[0],unit=unit,method=method);plt.plot(int1d.radial,int1d.intensity)
	plt.plot(int1d.radial,int1d.intensity-baseline)
	int1d=ai.integrate1d(img.data-isotropic,npt=npts[0],unit=unit,method=method);plt.plot(int1d.radial,int1d.intensity,'k--')
	plt.plot(mfilt1d.radial,mfilt1d.intensity)
	plt.plot(mfilt1d.radial,baseline)
	if 'SAXS' in filename:
		plt.yscale('log');plt.ylim([SAXSmin,None])
	plt.savefig(filename+'_iso1d.png')

	for isub in ['','_isub']:
		#2D
		plt.close('all')
		mpl.rc('text',usetex=True);mpl.rc('text.latex',preamble=r'\usepackage[helvet]{sfmath}')
		int2d=ai.integrate2d(img.data,npts[0],npts[1],unit=units,method=method)

		if geom=='Faser':
			plt.figure(figsize=(5.3/2.54,5.3/2.54))
			radg,azig=np.meshgrid(int2d.radial,int2d.azimuthal);args=np.where(abs(radg)==np.min(abs(radg)))
			int2d.intensity[args]=np.median([int2d.intensity[args[0],args[1]-1],int2d.intensity[args[0],args[1]+1]],axis=0)
			plt.gca().set_aspect('equal')
		else:
			plt.figure(figsize=(7.5/2.54,5.3/2.54))

		plt.pcolormesh(int2d.radial,int2d.azimuthal,int2d.intensity,cmap='coolwarm',norm=mpl.colors.LogNorm(vmin=SAXSmin
		) if 'SAXS' in filename else None)

		plt.xlabel(label(int2d.radial_unit));plt.ylabel(label(int2d.azimuthal_unit))
		plt.tick_params(axis='both',pad=2,labelsize=8)
		plt.tight_layout(pad=0.1)
		plt.savefig(filename+isub+'.png',dpi=300)

		#1D
		for azirang in [None,]:													####
			for radrang in [None,(20,21)]:										####
				plt.close('all')
				plt.figure(figsize=(7.5/2.54,5.3/2.54))
				mpl.rc('text',usetex=True);mpl.rc('text.latex',preamble=r'\usepackage[helvet]{sfmath}')

				if geom=='Faser':
					int1d=ai.integrate_fiber(img.data,npt_output=npts[0],output_unit=units[0],integrated_unit=units[1],integrated_unit_range=radrang,method=method,filename=filename+'_'+str(rang).replace(', ','-')+isub+'.xy')
					int1d.intensity[np.where(abs(int1d.radial)==min(abs(int1d.radial)))]=0
				else:
					int1d=ai.integrate1d(img.data,npt=npts[0],azimuth_range=azirang,unit=units,method=method,filename=filename+'_'+str(azirang).replace(', ','-')+isub+'.xy')
					np.savetxt(filename+'_'+str(radrang).replace(', ','-')+isub+'_rad.xy',np.array(ai.integrate_radial(img.data,npt=min(npts),radial_range=radrang,radial_unit=int2d.radial_unit,method=method)).transpose())

				plt.plot(int1d.radial,int1d.intensity,'k',linewidth=0.5)

				if 'SAXS' in filename:
					plt.yscale('log');plt.ylim([SAXSmin,None])
				plt.xlabel(label(int1d.unit));plt.ylabel(r'$I/1$')
				plt.tick_params(axis='both',pad=2,labelsize=8)
				plt.tight_layout(pad=0.1)
				plt.savefig(filename+'_'+str(azirang).replace(', ','-')+isub+'.png',dpi=300)

		img.data=img.data-isotropic
