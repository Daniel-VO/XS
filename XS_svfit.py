"""
Created 03. Dezember 2025 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import ray
import sys
import glob
import funcy
import lmfit as lm
import numpy as np
import sasmodels as sm
import matplotlib as mpl
import matplotlib.pyplot as plt
from sasmodels.core import load_model
from sasmodels.resolution import Slit1D
from sasmodels.direct_model import call_kernel

def pad(q):
	return np.concatenate((np.linspace(min(q)/2,min(q)),q,np.linspace(max(q),max(q)*2)))

def fitfunc(params):
	prm=params.valuesdict()
	global res
	res=((yobsS-SAXSres.apply(call_kernel(SAXSkernel,funcy.omit(prm,['Uscale','Ubackground'])))[50:-50])*qS**1)
	if os.path.isfile(f.replace('_SAXS','_USAXS')):
		prm['scale']=prm['Uscale'];prm['background']=prm['Ubackground']
		res=np.append(((yobsU-USAXSres.apply(call_kernel(USAXSkernel,funcy.omit(prm,['Uscale','Ubackground'])))[50:-50])*qU**1),res)
	return np.nan_to_num(res)

f=10;s=10;lamb=1.5406;rGon=300;d1=173.5;d2=2*rGon-d1
AHlen=(f/2+s/2)/d1*2*rGon
LHlen=(f/2+s/2)/d1*d2
a=4*np.pi*np.sin(np.arctan(AHlen/2/rGon))/lamb
b=4*np.pi*np.sin(np.arctan(LHlen/2/rGon))/lamb
dIW=4*np.pi*np.sin(np.arctan((AHlen-LHlen)/2/rGon))/lamb

model=load_model(str(sys.argv[1]))
params=lm.Parameters()
if str(sys.argv[1])=='unified_power_Rg':
	params.add('scale',1,min=0)
	params.add('background',0.001,min=0)
	params.add('level',int(sys.argv[2]),vary=False)
	for level in range(int(sys.argv[2])):
		params.add('rg'+str(level+1),15.8,min=0)
		params.add('power'+str(level+1),4,min=1,max=6)
		params.add('B'+str(level+1),4.5e-6,min=0)
		params.add('G'+str(level+1),400,min=0)
else:
	for item in model.info.parameters._get_defaults().items():
		if 'sld' not in item[0] and item[1]!=0 and item[1]!=90:
			params.add(item[0],item[1],min=0)

# ~ params.add('radius_pd',0,min=0,max=1)
# ~ params.add('radius_pd_n',10,vary=False)

os.system('mv '+model.info.id+'.log '+model.info.id+'.alt')

@ray.remote
def fit(g):
	filename=g.split('_SAXS')[0]
	global f,qS,yobsS,SAXSres,SAXSkernel;f=g
	qS,yobsS=np.genfromtxt(f,unpack=True)
	q_widthS=float(open(f).readlines()[0].split('=')[-1])

	# ~ qS,yobsS=qS[qS>1e-2],yobsS[qS>1e-2]						####

	if os.path.isfile(f.replace('_SAXS','_USAXS')):
		params.add('Uscale',params.valuesdict()['scale']/10,min=0);params.add('Ubackground',0,min=0)
		global qU,yobsU,USAXSres,USAXSkernel
		qU,yobsU=np.genfromtxt(f.replace('_SAXS','_USAXS'),unpack=True)
		q_widthU=float(open(f.replace('_SAXS','_USAXS')).readlines()[0].split('=')[-1])

		# ~ qU,yobsU=qU[qU>1e-3],yobsU[qU>1e-3]					####

		qUkernel=pad(qU)
		USAXSres=Slit1D(qUkernel,q_length=dIW,q_width=q_widthU,q_calc=qUkernel)
		USAXSkernel=model.make_kernel([qUkernel])

	qSkernel=pad(qS)
	SAXSres=Slit1D(qSkernel,q_length=dIW,q_width=q_widthS,q_calc=qSkernel)
	SAXSkernel=model.make_kernel([qSkernel])

	result=lm.minimize(fitfunc,params,method='leastsq')
	print(filename,sys.argv[1])
	result.params.pretty_print()
	prm=result.params.valuesdict()

	plt.close('all')
	mpl.rc('text',usetex=True)
	mpl.rc('text.latex',preamble=r'\usepackage[helvet]{sfmath}')
	plt.figure(figsize=(7.5/2.54,5.3/2.54))

	plt.plot(qS,yobsS,'.',ms=1,color='k')
	plt.plot(qSkernel[50:-50],SAXSres.apply(call_kernel(SAXSkernel,funcy.omit(prm,['Uscale','Ubackground'])))[50:-50],'0.3',linewidth=0.5)

	if os.path.isfile(f.replace('_SAXS','_USAXS')):
		prm['scale']=prm['Uscale'];prm['background']=prm['Ubackground']
		plt.plot(qU,yobsU,'.',color='k')
		plt.plot(qUkernel[50:-50],USAXSres.apply(call_kernel(USAXSkernel,funcy.omit(prm,['Uscale','Ubackground'])))[50:-50],'0.3',linewidth=0.5)

	plt.xscale('log');plt.yscale('log')

	plt.xlabel(r'$q/\rm{\AA}^{-1}$',fontsize=10)
	plt.ylabel(r'$I/1$',fontsize=10)
	plt.tick_params(axis='both',pad=2,labelsize=8)
	plt.tight_layout(pad=0.1)
	plt.savefig(filename+'_'+model.info.id+'.png',dpi=300)

	# ~ plt.close('all')
	# ~ plt.plot(qU,res[:len(qU)])
	# ~ plt.plot(qS,res[len(qU):])
	# ~ plt.xscale('log')
	# ~ plt.savefig(filename+'_'+model.info.id+'_res.png',dpi=300)

	print(filename,[str((param.name,'=',param.value,'+-',param.stderr)) for param in result.params.values()],np.sum(res**2),file=open(model.info.id+'.log','a'))

	return filename,result

np.save(model.info.id+'.npy',ray.get([fit.remote(g) for g in glob.glob('*_SAXS*.dat')]))
