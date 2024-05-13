"""
Created 13. May 2024 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import ray
import sys
import glob
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
	res=((yobsS-SAXSres.apply(call_kernel(SAXSkernel,prm))[50:-50])*qS**2)
	if os.path.isfile(f.replace('_RSAXS','_USAXS')):
		prm['scale']=prm['Uscale'];prm['background']=prm['Ubackground']
		res=np.append(((yobsU-USAXSres.apply(call_kernel(USAXSkernel,prm))[50:-50])*qU**2),res)
	return np.nan_to_num(res)

model=load_model(str(sys.argv[1]))
params=lm.Parameters()
if str(sys.argv[1])=='unified_power_Rg':
	params.add('scale',1,min=0)
	params.add('level',int(sys.argv[2]),vary=False)
	for level in np.arange(int(sys.argv[2])):
		params.add('rg'+str(level+1),15.8,min=0)
		params.add('power'+str(level+1),4,min=2,max=6)
		params.add('B'+str(level+1),4.5e-6,min=0)
		params.add('G'+str(level+1),400,min=0)
else:
	for item in model.info.parameters._get_defaults().items():
		params.add(item[0],item[1],min=0)

os.system('mv '+model.info.id+'.log '+model.info.id+'.alt')

@ray.remote
def fit(g):
	filename=g.split('_SAXS')[0]
	global f,qS,yobsS,SAXSres,SAXSkernel;f=g
	qS,yobsS=np.genfromtxt(f,unpack=True)
	qy_widthS=float(open(f).readlines()[0].split('=')[-1])

	# ~ qS,yobsS=qS[np.where(qS>1e-2)],yobsS[np.where(qS>1e-2)]						####

	if os.path.isfile(f.replace('_SAXS','_USAXS')):
		params.add('Uscale',params.valuesdict()['scale']/10,min=0);params.add('Ubackground',0,min=0)
		global qU,yobsU,USAXSres,USAXSkernel
		qU,yobsU=np.genfromtxt(f.replace('_SAXS','_USAXS'),unpack=True)
		qy_widthU=float(open(f.replace('_SAXS','_USAXS')).readlines()[0].split('=')[-1])

		# ~ qU,yobsU=qU[np.where(qU>1e-3)],yobsU[np.where(qU>1e-3)]					####

		qUkernel=pad(qU)
		USAXSres=Slit1D(qUkernel,qx_width=0.136,qy_width=qy_widthU,q_calc=qUkernel)
		USAXSkernel=model.make_kernel([qUkernel])

	qSkernel=pad(qS)
	SAXSres=Slit1D(qSkernel,qx_width=0.136,qy_width=qy_widthS,q_calc=qSkernel)
	SAXSkernel=model.make_kernel([qSkernel])

	result=lm.minimize(fitfunc,params,method='least_squares')
	print(filename,sys.argv[1])
	result.params.pretty_print()
	prm=result.params.valuesdict()

	plt.close('all')
	mpl.rc('text',usetex=True)
	mpl.rc('text.latex',preamble=r'\usepackage[helvet]{sfmath}')
	plt.figure(figsize=(7.5/2.54,5.3/2.54))

	plt.scatter(qS,yobsS,c='k',marker='.',s=2,linewidth=0)
	plt.plot(qSkernel[50:-50],SAXSres.apply(call_kernel(SAXSkernel,prm))[50:-50],'0.3',linewidth=0.5)

	if os.path.isfile(f.replace('_SAXS','_USAXS')):
		prm['scale']=prm['Uscale'];prm['background']=prm['Ubackground']
		plt.scatter(qU,yobsU,c='k',marker='.',s=2,linewidth=0)
		plt.plot(qUkernel[50:-50],USAXSres.apply(call_kernel(USAXSkernel,prm))[50:-50],'0.3',linewidth=0.5)

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

	print(filename,[str((param.name,'=',param.value,'+-',param.stderr)) for param in result.params.values()],file=open(model.info.id+'.log','a'))

	return filename,result

np.save(model.info.id+'.npy',ray.get([fit.remote(g) for g in glob.glob('*_SAXS*.dat')]))
