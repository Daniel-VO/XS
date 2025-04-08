"""
Created 08. April 2025 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import ray
import glob
import scipy
import numpy as np
import matplotlib.pyplot as plt

def toq(tt):
	return 4*np.pi*np.sin(np.radians(tt/2))/1.5406

def gaussian(x,a,x0,sigma):
	return a*np.exp(-((x-x0)/sigma)**2/2)

def gaussfit(q,y):
	args=np.where(q<=-min(q))
	params,_=scipy.optimize.curve_fit(gaussian,q[args],y[args],p0=[max(y),0,1e-3])
	return params

@ray.remote
def subt(f,bgfiles):
	filename=os.path.splitext(f)[0]
	tt,yobs,eps=np.genfromtxt((i.replace('*','#') for i in open(f)),unpack=True)
	q=toq(tt)																	#to q

	if len(bgfiles)==0:
		print('Kein Untergrund')
		HWHM=0
		ybg=np.zeros(yobs.shape)
	else:
		bgstrings=['Mylar']
		for bgstring in bgstrings:
			if bgstring in filename:
				bgfile=bgfiles[[i for i,l in enumerate(bgfiles) if bgstring in l][0]]
			else:
				bgfile=bgfiles[[i for i,l in enumerate(bgfiles) if bgstring not in l][0]]

		ttbg,ybg,epsbg=np.genfromtxt((i.replace('*','#') for i in open(bgfile)),unpack=True)
		qbg=toq(ttbg)															#to q
		if s=='*_USAXS' or s=='*_SAXS':
			gpm=gaussfit(q,yobs);gpb=gaussfit(qbg,ybg)
			q-=gpm[1];qbg-=gpb[1]												#zdc
			HWHM=(2*np.log(2))**0.5*abs(gpb[2])									#HWHM
		elif s=='*_RSAXS':
			HWHM=qbg[np.where(ybg<max(ybg)/2)[0][0]]
		else:
			HWHM=0

		argscut=np.where((q>=min(qbg))&(q<=max(qbg)))
		q=q[argscut];yobs=yobs[argscut]											#cut
		ybg=scipy.interpolate.interp1d(qbg,ybg)(q)
		yobs-=ybg/max(ybg)*max(yobs)											#bgcorr

		plt.close('all')
		plt.plot(q,yobs+ybg);plt.plot(q,ybg);plt.plot(q,yobs)
		plt.plot(np.linspace(q[0],-q[0]),gaussian(np.linspace(q[0],-q[0]),gpm[0],0,gpm[2]))
		plt.plot(np.linspace(q[0],-q[0]),gaussian(np.linspace(q[0],-q[0]),gpb[0],0,gpb[2]))
		plt.yscale('log');plt.xlim([q[0]*1.02,-q[0]*1.02]);plt.ylim([(yobs+ybg)[0]/1.02,max(gaussian(np.linspace(q[0],-q[0]),*gpb))*1.02])
		plt.savefig(filename+'_cb.png')

	plt.close('all')
	plt.plot(q,yobs)
	if len(bgfiles)!=0:
		plt.plot(q,yobs+ybg);plt.plot(q,ybg)
		plt.xlim([[1e-4,1e-3,1e-2,1e-3][int(np.where(np.array(['*_USAXS','*_SAXS','*_TXRD','*_RSAXS'])==s)[0][0])],None])
		plt.figtext(0.98,0.98,'BG: '+os.path.splitext(bgfile)[0],ha='right',va='top')
	plt.xscale('log');plt.yscale('log')
	plt.tight_layout(pad=0.1)
	plt.savefig(filename+'.png')

	with open(filename+'_bgs_toq.dat','a') as d:
		d.write('#qy_width = '+str(HWHM)+'\n')
		np.savetxt(d,np.transpose([q,yobs]),fmt='%.8f')

paths=glob.glob('*/')
paths.append('')

for p in paths:
	os.system('rm '+p+'*.dat '+p+'*.png')
	for s in ['*_USAXS','*_SAXS','*_TXRD','*_RSAXS']:
		ray.get([subt.remote(f,glob.glob(p+s+'*_BG.ras')) for f in glob.glob(p+s+'.ras')])
