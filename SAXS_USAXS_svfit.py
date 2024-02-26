"""
Created 14. February 2023 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
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

def fitfunc(params):
	prm=params.valuesdict()
	global res
	res=yobsS-SAXSres.apply(call_kernel(SAXSkernel,prm))
	if os.path.isfile(f.replace('_SAXS','_USAXS')):
		prm['scale']=prm['Uscale'];prm['background']=prm['Ubackground']
		res=np.append(yobsU-USAXSres.apply(call_kernel(USAXSkernel,prm)),res)
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
params.add('background',0)

os.system('mv '+model.info.id+'.log '+model.info.id+'.alt')
logfile=open(model.info.id+'.log','a')
res_collect=0
for f in glob.glob('*_SAXS*.dat'):
	filename=f.split('_SAXS')[0]
	plt.close('all')
	mpl.rc('text',usetex=True)
	mpl.rc('text.latex',preamble=r'\usepackage[helvet]{sfmath}')
	plt.figure(figsize=(7.5/2.54,5.3/2.54))

	qS,yobsS=np.genfromtxt(f,unpack=True)
	qy_width=float(open(f).readlines()[0].split('=')[-1])
	plt.figtext(0.98,0.98,'SAXS qy_width:\n'+str(round(qy_width,6)),fontsize=6,ha='right',va='top')
	qS,yobsS=qS[:np.argmin(yobsS)],yobsS[:np.argmin(yobsS)]	####
	SAXSres=Slit1D(qS,qx_width=0.136,qy_width=qy_width,q_calc=qS)
	SAXSkernel=model.make_kernel([qS])

	if os.path.isfile(f.replace('_SAXS','_USAXS')):
		qU,yobsU=np.genfromtxt(f.replace('_SAXS','_USAXS'),unpack=True)
		qy_width=float(open(f.replace('_SAXS','_USAXS')).readlines()[0].split('=')[-1])
		plt.figtext(0.2,0.23,'USAXS qy_width:\n'+str(round(qy_width,6)),fontsize=6)
		USAXSres=Slit1D(qU,qx_width=0.136,qy_width=qy_width,q_calc=qU)
		USAXSkernel=model.make_kernel([qU])
		params.add('Uscale',params.valuesdict()['scale']/10,min=0);params.add('Ubackground',0)

	result=lm.minimize(fitfunc,params)
	print(filename,sys.argv[1])
	result.params.pretty_print()
	prm=result.params.valuesdict()
	res_collect+=np.sum(res)

	plt.scatter(qS,yobsS,c='k',marker='.',s=2,linewidth=0)
	plt.plot(qS,SAXSres.apply(call_kernel(SAXSkernel,prm)),'0.3',linewidth=0.5)

	if os.path.isfile(f.replace('_SAXS','_USAXS')):
		prm['scale']=prm['Uscale'];prm['background']=prm['Ubackground']
		plt.scatter(qU,yobsU,c='k',marker='.',s=2,linewidth=0)
		plt.plot(qU,USAXSres.apply(call_kernel(USAXSkernel,prm)),'0.3',linewidth=0.5)

	plt.xscale('log');plt.yscale('log')

	plt.xlabel(r'$q/\rm{\AA}^{-1}$',fontsize=10)
	plt.ylabel(r'$I/1$',fontsize=10)
	plt.tick_params(axis='both',pad=2,labelsize=8)
	plt.tight_layout(pad=0.1)
	plt.savefig(filename+'_'+model.info.id+'.png',dpi=300)

	logfile.write(filename+' ')
	for param in result.params.values():
		logfile.write(str((param.name,'=',param.value,'+-',param.stderr)))
	logfile.write('\n')

logfile.write(str(res_collect))

