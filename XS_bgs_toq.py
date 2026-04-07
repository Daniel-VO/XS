"""
Created 27. Maerz 2026 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import ray
import glob
import scipy
import numpy as np
import matplotlib.pyplot as plt

def toq(tt_deg):
	return 4*np.pi*np.sin(np.radians(tt_deg/2))/1.5406

def gaussian(x,a,x0,sigma):
	return a*np.exp(-((x-x0)/sigma)**2/2)

def gaussfit(q,y):
	args=q<=-min(q)
	params,_=scipy.optimize.curve_fit(gaussian,q[args],y[args],p0=[max(y),0,1e-3])
	return params

os.system('rm *_toq.png')
os.system('rm *_toq.dat')

@ray.remote
def corr(f,bgfiles):
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
	d2=2*rGon-d1
	AHlen=(fs/2+ss/2)/d1*2*rGon
	LHlen=(fs/2+ss/2)/d1*d2
	a=4*np.pi*np.sin(np.arctan(AHlen/2/rGon))/lamb
	b=4*np.pi*np.sin(np.arctan(LHlen/2/rGon))/lamb
	dIW=4*np.pi*np.sin(np.arctan((AHlen-LHlen)/2/rGon))/lamb

	tt_deg,yobs,eps=np.genfromtxt((i.replace('*','#') for i in open(f)),unpack=True)
	yobs/=1+np.cos(np.radians(tt_deg))**2
	q=toq(tt_deg);bgs=''														#to q

	if len(bgfiles)==0:
		print('Kein Untergrund')
		HWHM=0
		ybg=np.zeros(yobs.shape)
	else:
		bgstrings=['_SAXS','_USAXS']
		for bgstring in bgstrings:
			if bgstring in fn:
				bgfile=bgfiles[[i for i,l in enumerate(bgfiles) if bgstring in l][0]]
			else:
				bgfile=bgfiles[[i for i,l in enumerate(bgfiles) if bgstring not in l][0]]

		ttbg_deg,ybg,epsbg=np.genfromtxt((i.replace('*','#') for i in open(bgfile)),unpack=True)
		ybg/=1+np.cos(np.radians(ttbg_deg))**2
		qbg=toq(ttbg_deg)														#to q
		if min(q)<0:
			gpm=gaussfit(q,yobs);gpb=gaussfit(qbg,ybg)
			q-=gpm[1];qbg-=gpb[1]												#zdc
			bxw=(2*np.log(2))**0.5*abs(gpb[2])									#HWHM
		else:
			bxw=qbg[np.where(ybg<max(ybg)/2)[0][0]]

		argscut=(q>=min(qbg))&(q<=max(qbg))
		q=q[argscut];yobs=yobs[argscut]											#cut
		ybg=scipy.interpolate.interp1d(qbg,ybg)(q)
		yobs-=ybg/gpb[0]*gpm[0];bgs='_bgs'										#bgcorr

		plt.close('all')
		plt.plot(q,yobs+ybg);plt.plot(q,ybg);plt.plot(q,yobs)
		plt.plot(np.linspace(q[0],-q[0]),gaussian(np.linspace(q[0],-q[0]),gpm[0],0,gpm[2]))
		plt.plot(np.linspace(q[0],-q[0]),gaussian(np.linspace(q[0],-q[0]),gpb[0],0,gpb[2]))
		plt.yscale('log');plt.xlim([q[0]*1.02,-q[0]*1.02]);plt.ylim([(yobs+ybg)[0]/1.02,max(gaussian(np.linspace(q[0],-q[0]),*gpb))*1.02])
		plt.savefig(fn+'_cb.png')

	plt.close('all')
	plt.plot(q,yobs)
	if len(bgfiles)!=0:
		plt.plot(q,yobs+ybg);plt.plot(q,ybg)
		plt.figtext(0.98,0.98,'BG: '+os.path.splitext(bgfile)[0],ha='right',va='top')
	plt.xscale('log');plt.yscale('log')
	plt.tight_layout(pad=0.1)
	plt.savefig(fn+'.png')

	with open(fn+bgs+'_toq.dat','a') as d:
		print('#'+str({'a':a,'b':b,'bxw':bxw,'dIW':dIW}),file=d)
		np.savetxt(d,np.transpose([q,yobs]),fmt='%.8f')

bgfiles=glob.glob('*_BG.ras')
ray.get([corr.remote(f,bgfiles) for f in glob.glob('*.ras')])
