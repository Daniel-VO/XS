"""
Created 27. Maerz 2026 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import ray
import sys
import glob
import scipy
import numpy as np
import matplotlib.pyplot as plt

def gaussian(x,a,x0,sigma):
	return a*np.exp(-((x-x0)/sigma)**2/2)

def gaussfit(q,y):
	args=q<=-min(q)
	params,_=scipy.optimize.curve_fit(gaussian,q[args],y[args],p0=[max(y),0,1e-3])
	return params

Mleer=input('Haltermessung, .ras: ')

ttbg_deg,ybg,epsbg=np.genfromtxt((i.replace('*','#') for i in open(Mleer)),unpack=True)

@ray.remote
def corr(f):
	fn=os.path.splitext(f)[0]

	for line in open(f,'r').readlines():
		if '*HW_XG_WAVE_LENGTH_ALPHA1' in line:
			lamb=float(eval(line.split(' ')[-1]))
		if '*HW_GONIOMETER_RADIUS-3' in line:
			rGon=float(eval(line.split(' ')[-1]))
		if '*HW_GONIOMETER_RADIUS-2' in line:
			d1=float(eval(line.split(' ')[-1]))
		if '*HW_XG_FOCUS "' in line:
			fs=float(eval(line.split('x')[-1].split('mm')[0]))
		if '"LLS"' in line:
			llsnumber=eval(line.split(' ')[0].split('-')[-1])
	for line in open(f,'r').readlines():
		if '*MEAS_COND_AXIS_POSITION-'+str(llsnumber) in line:
			ss=float(eval(line.split(' ')[-1]).split('mm')[0])

	AHlen,LHlen=(fs/2+ss/2)/d1*np.array([2*rGon,2*rGon-d1])
	a,b,dIW=4*np.pi*np.sin(np.arctan(np.array([AHlen,LHlen,AHlen-LHlen])/2/rGon))/lamb

	tt_deg,yobs,eps=np.genfromtxt((i.replace('*','#') for i in open(f)),unpack=True)
	global ttbg_deg,ybg
	q,qbg=4*np.pi*np.sin(np.radians(np.array([tt_deg,ttbg_deg])/2))/lamb		#to q

	if min(q)<0:
		gpm=gaussfit(q,yobs);gpb=gaussfit(qbg,ybg)
		q-=gpm[1];qbg-=gpb[1]													#zdc
		bxw=(2*np.log(2))**0.5*abs(gpb[2])										#HWHM
	else:
		bxw=qbg[np.where(ybg<max(ybg)/2)[0][0]]

	argscut=(q>=min(qbg))&(q<=max(qbg))
	q=q[argscut];yobs=yobs[argscut]												#cut
	ybg=scipy.interpolate.interp1d(qbg,ybg)(q)
	yobs-=ybg/gpb[0]*gpm[0]														#bgcorr

	yobs/=(1+np.cos(2*np.radians(tt_deg[argscut]/2))**2)/2						#polcorr

	plt.close('all')
	plt.plot(q,yobs)
	plt.xscale('log');plt.yscale('log')
	plt.tight_layout(pad=0.1)
	plt.savefig(fn+'.png')

	os.system('rm *_toq.dat')
	with open(fn+'_bgs_toq.dat','a') as d:
		print('#'+str({'a':a,'b':b,'bxw':bxw,'dIW':dIW}),file=d)
		np.savetxt(d,np.transpose([q,yobs]),fmt='%.8f')

ray.get([corr.remote(f) for f in glob.glob('*.ras')])
