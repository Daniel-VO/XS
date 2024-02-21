"""
Created 14. February 2023 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import glob
import lmfit as lm
import numpy as np
import sasmodels as sm
import matplotlib.pyplot as plt
from scipy import interpolate
from sasmodels.core import load_model
from sasmodels.data import Data1D
from sasmodels.direct_model import DirectModel,call_kernel
from sasmodels.resolution import Slit1D
from scipy import interpolate

def fitfunc(params):
	prm=params.valuesdict()
	res=SAXSdata.y-SAXSres.apply(call_kernel(SAXSkernel,prm))
	if os.path.isfile(f.replace('_SAXS','_USAXS')):
		prm['scale']=prm['Uscale'];prm['background']=prm['Ubackground']
		res=np.append(res,USAXSdata.y-USAXSres.apply(call_kernel(USAXSkernel,prm)))
	# ~ plt.plot(SAXSdata.x,SAXSdata.y)
	# ~ plt.plot(SAXSdata.x,SAXSres.apply(call_kernel(SAXSkernel,prm)))
	# ~ plt.plot(USAXSdata.x,USAXSdata.y)
	# ~ plt.plot(USAXSdata.x,USAXSres.apply(call_kernel(USAXSkernel,prm)))
	# ~ plt.xscale('log');plt.yscale('log')
	# ~ plt.show()
	# ~ print(sum(res))
	return abs(res)**0.5

model=load_model('unified_power_Rg')
params=lm.Parameters()
params.add('scale',1,min=0)
params.add('background',0)
params.add('level',3,vary=False)

params.add('rg1',1000,min=100,max=1500)
params.add('power1',4,min=2,max=6)
params.add('B1',4.5e-6,min=0)
params.add('G1',400,min=0)

params.add('rg2',100,min=10,max=1000)
params.add('power2',4,min=2,max=6)
params.add('B2',4.5e-6,min=0)
params.add('G2',400,min=0)

params.add('rg3',10,min=1,max=100)
params.add('power3',4,min=2,max=6)
params.add('B3',4.5e-6,min=0)
params.add('G3',400,min=0)

os.system('mv results.log results.alt')
for f in glob.glob('*_SAXS*.dat'):
	filename=f.split('_SAXS')[0]
	plt.close('all')

	q,yobs=np.genfromtxt(f,unpack=True)
	SAXSdata=Data1D(q,yobs,np.zeros(q.shape),np.zeros(yobs.shape))
	SAXSres=Slit1D(q,qx_width=0.136,qy_width=0.010,q_calc=q)
	SAXSkernel=model.make_kernel([q])

	if os.path.isfile(f.replace('_SAXS','_USAXS')):
		q,yobs=np.genfromtxt(f.replace('_SAXS','_USAXS'),unpack=True)
		USAXSdata=Data1D(q,yobs,np.zeros(q.shape),np.zeros(yobs.shape))
		USAXSres=Slit1D(q,qx_width=0.136,qy_width=0.000,q_calc=q)
		USAXSkernel=model.make_kernel([q])

		params.add('Uscale',1,min=0)
		params.add('Ubackground',0)

	result=lm.minimize(fitfunc,params)
	print(filename)
	result.params.pretty_print()
	prm=result.params.valuesdict()

	plt.plot(SAXSdata.x,SAXSdata.y)
	plt.plot(SAXSdata.x,SAXSres.apply(call_kernel(SAXSkernel,prm)))

	if os.path.isfile(f.replace('_SAXS','_USAXS')):
		prm['scale']=prm['Uscale'];prm['background']=prm['Ubackground']
		plt.plot(USAXSdata.x,USAXSdata.y)
		plt.plot(USAXSdata.x,USAXSres.apply(call_kernel(USAXSkernel,prm)))

	plt.xscale('log');plt.yscale('log')
	plt.savefig(filename+'_svfit.png')

	with open('results.log','a') as logfile:
		logfile.write(filename+' ')
		for param in result.params.values():
			logfile.write(str((param.name,'=',param.value,'+-',param.stderr)))
		logfile.write('\n')
