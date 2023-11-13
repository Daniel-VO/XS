"""
Created 13. November 2023 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import numpy as np
import os
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import fabio,pyFAI
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from pyFAI.io import DefaultAiWriter

def q_to_px(q):
	return np.tan(2*np.arcsin(q*wavelength/(4*np.pi)))*detdist/pxsizeX

def draw_azimuth(azim0,azim1):
	ax1.add_patch(mpl.patches.Wedge([beamcenterX,beamcenterY],max(img.shape)*2**0.5,-azim1,-azim0,color='k',alpha=0.1))
	plt.text(beamcenterX+img.shape[0]/3*np.cos(np.radians((azim1+azim0)/2)),beamcenterY-img.shape[0]/3*np.sin(np.radians((azim1+azim0)/2)),r'$\beta/^\circ:\rm{'+str(i[2])[1:-1]+'}$',ha='center',color='w',fontsize=8)

def draw_annulus(q0,q1):
	ax1.add_patch(mpl.patches.Annulus([beamcenterX,beamcenterY],q_to_px(q1),width=q_to_px(q1)-q_to_px(q0),color='k',alpha=0.1))
	plt.text(beamcenterX+q_to_px((q1+q0)/2)*img.shape[1]/img.shape[0]/2,beamcenterY-q_to_px((q1+q0)/2)*img.shape[0]/img.shape[1]/2**0.5,r'$q/\rm{\AA}:\rm{'+str(i[2])[1:-1]+'}$',ha='center',color='w',fontsize=8)

def take(headerkey,indices):
	return np.fromstring(img.header[headerkey],sep=' ')[indices]

bgimgpat='BG_image'
maxval=np.max([fabio.open(fi).data for fi in glob.glob('*[!'+bgimgpat+'].img')])

for f in glob.glob('*[!'+bgimgpat+'].img'):
	filename=os.path.splitext(f)[0];img=fabio.open(f)
	# ~ print(img.header);a=b
	detdist=take('PXD_GONIO_VALUES',-1);detsizeX,detsizeY=take('PXD_DETECTOR_SIZE',[0,1]);beamcenterX,beamcenterY,pxsizeX,pxsizeY=take('PXD_SPATIAL_DISTORTION_INFO',[0,1,2,3]);wavelength=take('SOURCE_WAVELENGTH',-1);omega,chi,phi=take('CRYSTAL_GONIO_VALUES',[0,1,2])
	ai=AzimuthalIntegrator(dist=detdist/1e3,poni1=beamcenterY*pxsizeY/1e3,poni2=beamcenterX*pxsizeX/1e3,pixel1=pxsizeY/1e3,pixel2=pxsizeX/1e3,wavelength=wavelength/1e10,rot1=0,rot2=0,rot3=0)
	# ~ print(ai)

	if len(glob.glob('*'+bgimgpat+'.img'))>0:
		dark=fabio.open(glob.glob('*'+bgimgpat+'.img')[0]).data/2
	else:
		dark=None

	unit='q_A^-1';azimints=[];radiints=[];plots=['azim','radi'];xlabels=[r'$q/\rm{\AA}$',r'$\beta/^\circ$'];masks=[]
	azimints.append((img.data,dark,(-180,180),unit))
	azimints.append((img.data,dark,(-10,10),unit))
	azimints.append((img.data,dark,(80,100),unit))
	# ~ radiints.append((img.data,dark,(0.01,0.6),unit))
	radiints.append((img.data,dark,(0.02,0.06),unit))
	radiints.append((img.data,dark,(0.12,0.16),unit))
	# ~ radiints.append((img.data,dark,(1.03,1.17),unit))
	# ~ radiints.append((img.data,dark,(2.15,2.28),unit))
	# ~ radiints.append((img.data,dark,(0.36,0.40),unit))

	for p,plot in enumerate([azimints,radiints]):
		plt.close('all')
		mpl.rc('text',usetex=True)
		mpl.rc('text.latex',preamble=r'\usepackage[helvet]{sfmath}')
		fig,ax1=plt.subplots(figsize=(7.5/2.54,5.3/2.54))

		for i in plot:
			if plots[p]=='azim':
				x,I=ai.integrate1d(i[0],775/2,dark=i[1],azimuth_range=i[2],unit=i[3])
			elif plots[p]=='radi':
				x,I=ai.integrate_radial(i[0],360,dark=i[1],radial_range=i[2],radial_unit=i[3])
				ax1.set_xticks([-180,-90,0,90,180])
			# ~ DefaultAiWriter('',ai).save1D(filename+'_'+plots[p]+'_'+str(i[2])+'.dat',x,I,dim1_unit=i[3])
			ax1.plot(x,I/maxval,linewidth=1,label=xlabels[abs(p-1)]+r'$:\rm{'+str(i[2])[1:-1]+'}$')

		if 'SAXS' in filename:
			plt.yscale('log')
		plt.legend(frameon=False,fontsize=8)
		ax1.set_xlabel(xlabels[p],fontsize=10)
		ax1.set_ylabel(r'$I/1$',fontsize=10)
		ax1.tick_params(axis='both',pad=2,labelsize=8)
		plt.tight_layout(pad=0.1)
		plt.savefig(filename+'_'+plots[p]+'.png',dpi=300)

	####
	plt.close('all')
	mpl.rc('text',usetex=True)
	mpl.rc('text.latex',preamble=r'\usepackage[helvet]{sfmath}')
	fig,ax1=plt.subplots(figsize=(7.5*img.shape[1]/img.shape[0]/2.54,7.5/2.54))

	for i in azimints:
		if i[2]!=(-180,180) and i[2]!=(0,360):
			draw_azimuth(i[2][0],i[2][1])
	for i in radiints:
		draw_annulus(i[2][0],i[2][1])

	if 'SAXS' in filename:
		ax1.imshow(np.log(img.data+1),cmap='coolwarm',vmin=0,vmax=np.log(maxval))
	else:
		ax1.imshow(img.data,cmap='coolwarm',vmin=0,vmax=np.max(maxval))

	plt.axis('off');plt.tight_layout(pad=0)
	plt.savefig(filename+'.png',dpi=300)#,bbox_inches=mpl.transforms.Bbox([[4/2.54,0],[12/2.54,8/2.54]]))
