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

def profile(data,qlim,azilim,rbins,abins):
	y,x=np.indices((data.shape))
	xrel=x-data.shape[1]/2+0.5;yrel=y-data.shape[0]/2+0.5
	q0=4*np.pi*np.sin((xrel**2+yrel**2)**0.5*np.arctan(pxsizeX/detdist)/2)/wavelength;azi0=-np.degrees(np.angle(xrel+yrel*1j))
	args=np.where((q0>=qlim[0])&(q0<=qlim[1])&(azi0>=azilim[0])&(azi0<=azilim[1]))
	ints,q,azi=np.histogram2d(q0[args].flatten(),azi0[args].flatten(),weights=data[args].flatten(),bins=(rbins,abins))
	return q0,azi0,q[1:],azi[1:],ints

def center_pad(img,beamcenterX,beamcenterY):
		padx0=int((img.shape[1]/2-beamcenterX)*(np.sign(img.shape[1]/2-beamcenterX)+1))
		pady0=int((img.shape[0]/2-beamcenterY)*(np.sign(img.shape[0]/2-beamcenterY)+1))
		padx1=int((beamcenterX-img.shape[1]/2)*(np.sign(beamcenterX-img.shape[1]/2)+1))
		pady1=int((beamcenterY-img.shape[0]/2)*(np.sign(beamcenterY-img.shape[0]/2)+1))
		return np.pad(img.data,((pady0,pady1),(padx0,padx1)))

def take(img,headerkey,indices):
	return np.fromstring(img.header[headerkey],sep=' ')[indices]

for f in glob.glob('alle.img'):
	filename=os.path.splitext(f)[0].replace('_image','')
	img=fabio.open(f)
	img.data=img.data.astype(np.float32)/np.max(img.data)
	# ~ print(img.header,file=open('header','w'));a=b
	detdist=take(img,'PXD_GONIO_VALUES',-1);detsizeX,detsizeY=take(img,'PXD_DETECTOR_SIZE',[0,1]);beamcenterX,beamcenterY,pxsizeX,pxsizeY=take(img,'PXD_SPATIAL_DISTORTION_INFO',[0,1,2,3]);wavelength=take(img,'SOURCE_WAVELENGTH',-1);omega,chi,phi=take(img,'CRYSTAL_GONIO_VALUES',[0,1,2])

	if beamcenterX>1 and beamcenterY>1:
		img.data=center_pad(img,beamcenterX,beamcenterY)

	q0,azi0,q,azi,ints=profile(img.data,(0,np.inf),(-180,180),2,2)
	argsort=np.argsort(q0.flatten());argsort_inv=np.arange(len(argsort));argsort_inv[argsort]=argsort_inv.copy()
	img.data-=scipy.ndimage.median_filter(img.data.flatten()[argsort],size=len(q0.flatten())//500)[argsort_inv].reshape(img.data.shape)

	plt.imsave(filename+'.png',img.data,cmap='coolwarm')

	integrators=[]
	# ~ integrators.append([img.data,(0.1,0.7),(-180,180),120,1])
	# ~ integrators.append([img.data,(1,1.2),(-45,45),1,90])
	# ~ integrators.append([img.data,(0.4,2.7),(-30,30),115,1])

	for i in integrators:
		plt.close('all')
		mpl.rc('text',usetex=True)
		mpl.rc('text.latex',preamble=r'\usepackage[helvet]{sfmath}')
		plt.figure(figsize=(7.5/2.54,5.3/2.54))

		q0,azi0,q,azi,ints=profile(i[0],i[1],i[2],i[3],i[4])
		qlabel=r'$q/\rm{\AA}^{-1}:\rm{'+str(i[1])[1:-1]+'}$';betalabel=r'$\beta/^\circ:\rm{'+str(i[2])[1:-1]+'}$'

		if i[3]==1:
			plt.plot(azi,ints.flatten(),linewidth=1,label=qlabel)
			plt.xlabel(r'$\beta/^\circ$',fontsize=10);plt.ylabel(r'$I/1$',fontsize=10)
		elif i[4]==1:
			plt.plot(q,ints.flatten(),linewidth=1,label=betalabel)
			plt.xlabel(r'$q/\rm{\AA}^{-1}$',fontsize=10);plt.ylabel(r'$I/1$',fontsize=10)
		else:
			plt.contourf(azi,q,ints,cmap='coolwarm')
			plt.xlabel(r'$\beta/^\circ$',fontsize=10);plt.ylabel(r'$q/\rm{\AA}^{-1}$',fontsize=10)

		plt.legend(frameon=False,fontsize=8)
		plt.tick_params(axis='both',pad=2,labelsize=8)
		plt.tight_layout(pad=0.1)
		plt.savefig(str(i[1:])+'.png',dpi=300)

	plt.close('all')
	mpl.rc('text',usetex=True)
	mpl.rc('text.latex',preamble=r'\usepackage[helvet]{sfmath}')
	plt.subplot(projection='polar')
	q0,azi0,q,azi,ints=profile(img.data,(0,np.inf),(-180,180),2,2)
	img.data[np.where(q0<0.078)]=0;plt.ylim([None,0.67])							####	np.max(q0)/(2**2+1**2)**0.5
	plt.pcolormesh(np.radians(azi0),q0,img.data,cmap='coolwarm',vmin=np.quantile(img.data,0.5),
																vmax=np.quantile(img.data,1-1e-4));plt.grid(True)
	for i in integrators:
		q0,azi0,q,azi,ints=profile(i[0],i[1],i[2],i[3]+100,i[4]+100)
		if abs(i[2][1]-i[2][0])!=360:
			plt.contourf(np.radians(azi),q,np.ones(ints.shape),colors='k',alpha=0.1)
	plt.xticks(plt.xticks()[0],[r'$'+str(np.degrees(ang))+'^\circ$' for ang in plt.xticks()[0]],fontsize=8)
	plt.gca().set_rlabel_position(-90)
	plt.tick_params(axis='both',pad=2,labelsize=8);plt.tight_layout(pad=0.1)
	plt.savefig(filename+'_intmap.png',dpi=300)

