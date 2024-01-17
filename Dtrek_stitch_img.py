"""
Created 17. Januar 2023 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import fabio

def profile(data,qlim,azilim,rbins,abins):
	y,x=np.indices((data.shape))
	xrel=x-data.shape[1]/2;yrel=y-data.shape[0]/2
	q0=4*np.pi*np.sin((xrel**2+yrel**2)**0.5*np.arctan(pxsizeX/detdist)/2)/wavelength;azi0=-np.degrees(np.angle(xrel+yrel*1j))
	args=np.where((q0>=qlim[0])&(q0<=qlim[1])&(azi0>=azilim[0])&(azi0<=azilim[1]))
	ints,q,azi=np.histogram2d(q0[args].flatten(),azi0[args].flatten(),weights=data[args].flatten(),bins=(rbins,abins))
	return q0,azi0,q[1:],azi[1:],ints

def take(headerkey,indices):
	return np.fromstring(img.header[headerkey],sep=' ')[indices]

stack=[]
for f in glob.glob('*.img'):
	img=fabio.open(f)
	# ~ print(img.header,file=open('header','w'));a=b
	detdist=take('PXD_GONIO_VALUES',-1);detsizeX,detsizeY=take('PXD_DETECTOR_SIZE',[0,1]);beamcenterX,beamcenterY,pxsizeX,pxsizeY=take('PXD_SPATIAL_DISTORTION_INFO',[0,1,2,3]);wavelength=take('SOURCE_WAVELENGTH',-1);omega,chi,phi=take('CRYSTAL_GONIO_VALUES',[0,1,2])
####
	ttmin,ttmax=take('SCAN_DET_ROTATION',[0,1])
	img.data=np.pad(img.data,((0,0),(int(img.shape[1]+2*ttmin*img.shape[1]/(ttmax-ttmin)),0)))
	fs=(img.shape[0]**2+img.shape[1]**2)**0.5
	img.data=scipy.ndimage.rotate(img.data,phi-90-chi)
	img.data=img.data[:img.shape[0]//2*2,:img.shape[1]//2*2]
	padx=int((fs-img.shape[0])/2);pady=int((fs-img.shape[1])/2)
	img.data=np.pad(img.data,((padx,padx),(pady,pady)))
	stack.append(img.data)
stack=np.max(stack,axis=0)/np.max(stack)
plt.imsave('stack.png',stack,cmap='coolwarm')
np.save('stack.npy',stack)
####

stack=np.load('stack.npy')

integrators=[]
integrators.append([stack,(0,3),(-45,45),150,45])
integrators.append([stack,(1,1.2),(-45,45),1,90])
integrators.append([stack,(0.4,2.7),(-30,30),115,1])

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
ylim=2.8
q0,azi0,q,azi,ints=profile(stack,(0,ylim),(-180,180),2,2);plt.ylim([None,ylim])
plt.pcolormesh(np.radians(azi0),q0,stack,cmap='coolwarm');plt.grid(True)
for i in integrators:
	q0,azi0,q,azi,ints=profile(i[0],i[1],i[2],i[3]+100,i[4]+100)
	plt.contourf(np.radians(azi),q,np.ones(ints.shape),colors='k',alpha=0.1)
plt.xticks(plt.xticks()[0],[r'$'+str(np.degrees(ang))+'^\circ$' for ang in plt.xticks()[0]],fontsize=8)
plt.gca().set_rlabel_position(-90)
plt.tick_params(axis='both',pad=2,labelsize=8);plt.tight_layout(pad=0.1)
plt.savefig('intmap.png',dpi=300)

# ~ fabio.dtrekimage.DtrekImage(data=stack,header=img.header).write('stack.img')
