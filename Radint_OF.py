"""
Created 10. June 2025 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import glob
import lmfit as lm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def OF(p0,p,phihkl,scale):	#Onsager distribution function
	return scale*(p0+(1-p0)*p*np.cosh(p*np.cos(phi-phihkl))/np.sinh(p)\
		  *2*np.pi/np.trapz(p*np.cosh(p*np.cos(phi-phihkl))/np.sinh(p),x=phi))

def fitfunc(params):
	prm=params.valuesdict()
	res=np.zeros(yobs.shape)
	for o in Orientierungen:
		res+=OF(prm['p0_'+str(o)],prm['p_'+str(o)],prm['phihkl_'+str(o)],prm['scale'])
	return np.nan_to_num(yobs-res)

os.system('mv OF.log OF.alt')

Orientierungen=np.arange(1)

for f in glob.glob('*_rad.xy'):
	filename=os.path.splitext(f)[0]
	phi_deg,yobs=np.genfromtxt(f,unpack=True)
	phi=np.radians(phi_deg)

	phicorr=phi-phi[np.argmax(yobs)]
	args=np.where((phicorr>=0)&(phicorr<=90))
	cosq=np.trapz((yobs*np.cos(phicorr)**2*np.sin(phicorr))[args],x=phicorr[args])/np.trapz((yobs*np.sin(phicorr))[args],x=phicorr[args])
	fc=(3*cosq-1)/2

	params=lm.Parameters()
	params.add('scale',min(yobs),min=0)
	for o in Orientierungen:
		params.add('p0_'+str(o),0.5,min=0,max=1)
		params.add('p_'+str(o),50,min=0)
		params.add('phihkl_'+str(o),phi[np.where(yobs==max(yobs))][0])

	result=lm.minimize(fitfunc,params,method='bfgs')
	prm=result.params.valuesdict()
	print(filename)
	result.params.pretty_print()

	print(filename,fc.round(4),phi_deg[np.argmax(yobs)].round(4),[str((param.name,'=',param.value,'+-',param.stderr)) for param in result.params.values()],file=open('OF.log','a'))

	plt.close('all')
	plt.figure(figsize=(7.5/2.54,5.3/2.54))
	mpl.rc('text',usetex=True);mpl.rc('text.latex',preamble=r'\usepackage[helvet]{sfmath}')

	plt.plot(phi_deg,yobs,'k',linewidth=0.5)
	plt.plot(phi_deg,       [OF(prm['p0_'+str(o)],prm['p_'+str(o)],prm['phihkl_'+str(o)],prm['scale']) for o in Orientierungen][0]     ,linewidth=0.5)
	plt.plot(phi_deg,np.sum([OF(prm['p0_'+str(o)],prm['p_'+str(o)],prm['phihkl_'+str(o)],prm['scale']) for o in Orientierungen],axis=0),linewidth=0.5)

	plt.xlabel(r'$\chi/^\circ$');plt.ylabel(r'$I/1$')
	plt.tick_params(axis='both',pad=2,labelsize=8)
	plt.tight_layout(pad=0.1)
	plt.savefig(filename+'_OF.png',dpi=300)
