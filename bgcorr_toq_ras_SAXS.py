"""
Created 14. February 2023 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

paths=glob.glob('*/')
paths.append('')

for p in paths:
	os.system('rm '+p+'*.dat '+p+'*.png')
	for s in ['*_SAXS','*_USAXS']:
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
			tt,yobs,yerr=np.genfromtxt((i.replace('*','#') for i in open(f)),unpack=True)
			if min(tt)<0:
				argsavg=np.where((tt>=min(tt))&(tt<=-min(tt)))
				tt-=np.average(tt[argsavg],weights=yobs[argsavg]**2)			#zero drift correction
			if len(BGfiles)==1:
				argscut=np.where((tt>=min(ttbg))&(tt<=max(ttbg)))
				tt=tt[argscut];yobs=yobs[argscut]
				ybg=bg(tt)
				ybg/=max(ybg);yobs/=max(yobs)
				yobs-=ybg													#bgcorr

			q=4*np.pi*np.sin(np.radians(tt/2))/1.5406							#toq
			mincoord=int(np.where(yobs==max(yobs[np.where(q>[5e-3,5e-4][int(np.where(np.array(['*_SAXS', '*_USAXS'])==s)[0])])]))[0])
			print('qmin = '+str(q[mincoord])+' A^-1')

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
			plt.xscale('log'),plt.yscale('log'),plt.xlim([1e-4,None])
			plt.savefig(filename+'.png')

			np.savetxt(filename+'_s_s_q.dat',np.transpose([q[mincoord:],yobs[mincoord:],np.zeros(q[mincoord:].shape)]),fmt='%.16f')
