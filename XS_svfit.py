"""
Created 27. Maerz 2025 by Daniel Van Opdenbosch, Technical University of Munich

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
def fit(f):
	fn=os.path.splitext(f)[0]
	q,yobs=np.genfromtxt(f,unpack=True)
	slinf=eval(open(f).readlines()[0].split('#')[-1])

	q,yobs=q[q>7e-3],yobs[q>7e-3]												####

	qpad=pad(q)
	smear=Slit1D(qpad,q_length=slinf['dIW'],q_width=slinf['bxw'],q_calc=qpad)
	kernel=model.make_kernel([qpad])

	def fitfunc(params):
		prm=params.valuesdict();global res
		res=((yobs-smear.apply(call_kernel(kernel,funcy.omit(prm,['Uscale','Ubackground'])))[50:-50])*q**1)
		return np.nan_to_num(res)

	result=lm.minimize(fitfunc,params,method='leastsq')
	print(fn,sys.argv[1])
	result.params.pretty_print()
	prm=result.params.valuesdict()

	plt.close('all')
	mpl.rc('text',usetex=True)
	mpl.rc('text.latex',preamble=r'\usepackage[helvet]{sfmath}')
	plt.figure(figsize=(7.5/2.54,5.3/2.54))

	plt.plot(q,yobs,'.',ms=1,color='k')
	plt.plot(qpad[50:-50],smear.apply(call_kernel(kernel,funcy.omit(prm,['Uscale','Ubackground'])))[50:-50],'0.3',linewidth=0.5)

	plt.xscale('log');plt.yscale('log')

	plt.xlabel(r'$q/\rm{\AA}^{-1}$',fontsize=10)
	plt.ylabel(r'$I/1$',fontsize=10)
	plt.tick_params(axis='both',pad=2,labelsize=8)
	plt.tight_layout(pad=0.1)
	plt.savefig(fn+'_'+model.info.id+'.png',dpi=300)

	if model.info.id=='unified_power_Rg':
		prm0=result.params
		result.params=lm.Parameters()
		lnums=sum(['rg' in k for k  in prm0.keys()])
		sortkey=np.argsort([prm0['rg'+str(level+1)] for level in range(lnums)])
		for level in range(lnums):
			result.params['rg'+str(level+1)]=prm0['rg'+str(sortkey[level]+1)]
			result.params['power'+str(level+1)]=prm0['power'+str(sortkey[level]+1)]
			result.params['B'+str(level+1)]=prm0['B'+str(sortkey[level]+1)]
			result.params['G'+str(level+1)]=prm0['G'+str(sortkey[level]+1)]

	print(fn,[str((param.name,'=',param.value,'+-',param.stderr)) for param in result.params.values()],np.sum(res**2),file=open(model.info.id+'.log','a'))

	return fn,result

np.save(model.info.id+'.npy',ray.get([fit.remote(f) for f in glob.glob('*.dat')]))
