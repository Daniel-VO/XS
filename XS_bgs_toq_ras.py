"""
Created 13. May 2024 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import glob
import scipy
import numpy as np
import matplotlib.pyplot as plt

def toq(tt):
	return 4*np.pi*np.sin(np.radians(tt/2))/1.5406

def zdc(q,y):
	amax=np.argmax(y)
	return q-np.average(q[0:2*amax+1],weights=y[0:2*amax+1]**2)

def gaussian(x,a,x0,sigma):
	return a*np.exp(-((x-x0)/sigma)**2/2)

paths=glob.glob('*/')
paths.append('')

for p in paths:
	os.system('rm '+p+'*.dat '+p+'*.png')
	for s in ['*_USAXS','*_SAXS','*_TXRD','*_RSAXS']:
		BGfiles=glob.glob(p+s+'*_BG.ras')

		if len(BGfiles)>1:
			print('Warnung: Mehr als ein Untergrund!')
		elif len(BGfiles)==1:
			print('bg: ',BGfiles[0])
			ttbg,ybg,epsbg=np.genfromtxt((i.replace('*','#') for i in open(BGfiles[0])),unpack=True)
			qbg=toq(ttbg)														#to q
			qbg=zdc(qbg,ybg)													#zero drift correction
			if s=='*_USAXS' or s=='*_SAXS':
				argsgauss=np.where((qbg>=min(qbg))&(qbg<=-min(qbg)))
				popt,pcov=scipy.optimize.curve_fit(gaussian,qbg[argsgauss],ybg[argsgauss],p0=[max(ybg),0,1e-4])
				HWHM=(2*np.log(2))**0.5*popt[-1]								#HWHM
			elif s=='*_RSAXS':
				HWHM=qbg[np.where(ybg<max(ybg)/2)[0][0]]
			else:
				HWHM=0
			print('qy_width = '+str(HWHM)+' A^-1')
			bg=scipy.interpolate.interp1d(qbg,ybg)								#bgint

		for f in glob.glob(p+s+'*[!_BG].ras'):
			filename=os.path.splitext(f)[0]
			tt,yobs,eps=np.genfromtxt((i.replace('*','#') for i in open(f)),unpack=True)
			q=toq(tt)															#to q
			q=zdc(q,yobs)														#zero drift correction
			argscut=np.where((q>=min(qbg))&(q<=max(qbg)))
			q=q[argscut];yobs=yobs[argscut]										#cut
			ybg=bg(q)
			yobs-=ybg															#bgcorr

			plt.close('all')
			if len(BGfiles)==1:
				plt.plot(q,yobs+ybg);plt.plot(q,ybg)
			plt.plot(q,yobs)
			plt.yscale('log'),plt.xlim([q[0]*1.02,-q[0]*1.02]),plt.ylim([(yobs+ybg)[0]/1.02,max(ybg)*1.02])
			plt.savefig(filename+'_cb.png')

			plt.close('all')
			if len(BGfiles)==1:
				plt.plot(q,yobs+ybg);plt.plot(q,ybg)
			plt.plot(q,yobs);plt.plot(q,yobs)
			plt.xscale('log'),plt.yscale('log'),plt.xlim([[1e-4,1e-3,1e-2,1e-3][int(np.where(np.array(['*_USAXS','*_SAXS','*_TXRD','*_RSAXS'])==s)[0][0])],None]),plt.ylim([None,2*max(yobs)])
			plt.savefig(filename+'.png')

			with open(filename+'_bgs_toq.dat','a') as d:
				d.write('#qy_width = '+str(HWHM)+'\n')
				np.savetxt(d,np.transpose([q,yobs]),fmt='%.16f')

