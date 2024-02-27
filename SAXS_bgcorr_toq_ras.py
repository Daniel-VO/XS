"""
Created 14. February 2023 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate,optimize

def gaussian(x,a,x0,sigma):
	return a*np.exp(-((x-x0)/sigma)**2/2)

paths=glob.glob('*/')
paths.append('')

for p in paths:
	os.system('rm '+p+'*.dat '+p+'*.png')
	for s in ['*_USAXS','*_SAXS']:
		BGfiles=glob.glob(p+s+'*BG.ras')

		if len(BGfiles)>1:
			print('Warnung: Mehr als ein Untergrund!')
		elif len(BGfiles)==1:
			ttbg,ybg,epsbg=np.genfromtxt((i.replace('*','#') for i in open(BGfiles[0])),unpack=True)
			argsavg=np.where((ttbg>=min(ttbg))&(ttbg<=-min(ttbg)))
			ttbg-=np.average(ttbg[argsavg],weights=ybg[argsavg]**2)		#zero drift correction
			bg=interpolate.interp1d(ttbg,ybg)

		for f in glob.glob('[!BG]'+p+s+'.ras'):
			filename=os.path.splitext(f)[0]
			print(filename)
			tt,yobs,eps=np.genfromtxt((i.replace('*','#') for i in open(f)),unpack=True)
			if min(tt)<0:
				argsavg=np.where((tt>=min(tt))&(tt<=-min(tt)))
				tt-=np.average(tt[argsavg],weights=yobs[argsavg]**2)			#zero drift correction
			if len(BGfiles)==1:
				argscut=np.where((tt>=min(ttbg))&(tt<=max(ttbg)))
				tt=tt[argscut];yobs=yobs[argscut]
				ybg=bg(tt)
				yobs-=ybg													#bgcorr

			q=4*np.pi*np.sin(np.radians(tt/2))/1.5406							#toq
			mincoord=int(np.where(yobs==max(yobs[np.where(q>[5e-4,5e-3][int(np.where(np.array(['*_USAXS','*_SAXS'])==s)[0][0])])]))[0][0])
			print('qmin = '+str(q[mincoord])+' A^-1')
			argsgauss=np.where((q>=q[0])&(q<=-q[0]))
			popt,pcov=optimize.curve_fit(gaussian,q[argsgauss],ybg[argsgauss],p0=[max(ybg),0,1e-4])
			HWHM=(2*np.log(2))**0.5*popt[-1]
			print('qy_width = '+str(HWHM)+' A^-1')

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
			plt.xscale('log'),plt.yscale('log'),plt.xlim([[1e-4,1e-3][int(np.where(np.array(['*_USAXS','*_SAXS'])==s)[0][0])],None])
			plt.savefig(filename+'.png')

			with open(filename+'_s_s_q.dat','a') as f:
				f.write('#qy_width = '+str(HWHM)+'\n')
				np.savetxt(f,np.transpose([q[mincoord:],yobs[mincoord:]]),fmt='%.16f')

