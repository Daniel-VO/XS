"""
Created 04. March 2024 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import glob
import scipy
import numpy as np
import matplotlib.pyplot as plt

def gaussian(x,a,x0,sigma):
	return a*np.exp(-((x-x0)/sigma)**2/2)

paths=glob.glob('*/')
paths.append('')

for p in paths:
	os.system('rm '+p+'*.dat '+p+'*.png')
	for s in ['*_USAXS','*_SAXS','*_TXRD']:
		BGfiles=glob.glob(p+s+'*BG.ras')

		if len(BGfiles)>1:
			print('Warnung: Mehr als ein Untergrund!')
		elif len(BGfiles)==1:
			ttbg,ybg,epsbg=np.genfromtxt((i.replace('*','#') for i in open(BGfiles[0])),unpack=True)
			qbg=4*np.pi*np.sin(np.radians(ttbg/2))/1.5406
			amaxbg=np.argmax(ybg)
			qbg-=np.average(qbg[0:2*amaxbg+1],weights=ybg[0:2*amaxbg+1]**2)		#zero drift correction
			argsgauss=np.where((qbg>=min(qbg))&(qbg<=-min(qbg)))
			popt,pcov=scipy.optimize.curve_fit(gaussian,qbg[argsgauss],ybg[argsgauss],p0=[max(ybg),0,1e-4])
			HWHM=(2*np.log(2))**0.5*popt[-1]									#HWHM
			bg=scipy.interpolate.interp1d(qbg,ybg)								#bgint

		for f in glob.glob('[!BG]'+p+s+'.ras'):
			filename=os.path.splitext(f)[0]
			print(filename)
			tt,yobs,eps=np.genfromtxt((i.replace('*','#') for i in open(f)),unpack=True)
			q=4*np.pi*np.sin(np.radians(tt/2))/1.5406
			amax=np.argmax(yobs)
			q-=np.average(q[0:2*amax+1],weights=yobs[0:2*amax+1]**2)			#zero drift correction
			argscut=np.where((q>=min(qbg))&(q<=max(qbg)))
			q=q[argscut];yobs=yobs[argscut]										#cut
			ybg=bg(q)
			yobs-=ybg															#bgcorr
			mincoord=np.where(scipy.signal.savgol_filter(yobs,3,1)<0)[-1][-1]+1
			print('qmin = '+str(q[mincoord])+' A^-1; qy_width = '+str(HWHM)+' A^-1')

			plt.close('all')
			if len(BGfiles)==1:
				plt.plot(q,yobs+ybg);plt.plot(q,ybg)
			plt.plot(q,yobs)
			plt.yscale('log'),plt.xlim([q[0]*1.02,-q[0]*1.02]),plt.ylim([(yobs+ybg)[0]/1.02,max(ybg)*1.02])
			plt.savefig(filename+'_cb.png')

			plt.close('all')
			if len(BGfiles)==1:
				plt.plot(q,yobs+ybg);plt.plot(q,ybg)
			plt.plot(q,yobs);plt.plot(q[mincoord:],yobs[mincoord:])
			plt.xscale('log'),plt.yscale('log'),plt.xlim([[5e-4,5e-3,5e-2][int(np.where(np.array(['*_USAXS','*_SAXS','*_TXRD'])==s)[0][0])],None]),plt.ylim([None,max(yobs*1.05)])
			plt.savefig(filename+'.png')

			with open(filename+'_s_s_q.dat','a') as f:
				f.write('#qy_width = '+str(HWHM)+'\n')
				np.savetxt(f,np.transpose([q[mincoord:],yobs[mincoord:]]),fmt='%.16f')

