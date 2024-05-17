import os
import glob
import lmfit
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def OF(p0,p,phihkl,scale):	#Onsager distribution function
	return scale*(p0+(1-p0)*p*np.cosh(p*np.cos(phi-phihkl))/np.sinh(p))

def fitfunc(params):
	prm=params.valuesdict()
	res=np.zeros(yobs.shape)
	for o in Orientierungen:
		res+=OF(prm['p0_'+str(o)],prm['p_'+str(o)],prm['phihkl_'+str(o)],prm['scale'])
	return np.nan_to_num(yobs-res)

os.system('mv OF.log OF.alt')

Orientierungen=np.arange(2)

for f in glob.glob('*radi.dat'):
	filename=os.path.splitext(f)[0]
	phi_deg,yobs=np.genfromtxt(f,unpack=True)
	phi=np.radians(phi_deg)

	args=np.where((phi_deg>=0)&(phi_deg<=180))
	cosq=np.trapz((yobs*np.cos(phi)**2*np.sin(phi))[args],x=phi[args])/np.trapz((yobs*np.sin(phi))[args],x=phi[args])
	fc=(3*cosq-1)/2

	params=lmfit.Parameters()
	params.add('scale',1,min=0)
	for o in Orientierungen:
		params.add('p0_'+str(o),0.5,min=0)
		params.add('p_'+str(o),0.1,min=0)
		params.add('phihkl_'+str(o),phi[np.where(yobs==max(yobs))][0])

	result=lmfit.minimize(fitfunc,params,method='bfgs')
	prm=result.params.valuesdict()
	print(filename)
	result.params.pretty_print()

	print(filename,fc,[str((param.name,'=',param.value,'+-',param.stderr)) for param in result.params.values()],file=open('OF.log','a'))

	plt.close('all')
	plt.plot(phi_deg,yobs)
	for o in Orientierungen:
		plt.plot(phi_deg,OF(prm['p0_'+str(o)],prm['p_'+str(o)],prm['phihkl_'+str(o)],prm['scale']))
	plt.plot(phi_deg,np.sum([OF(prm['p0_'+str(o)],prm['p_'+str(o)],prm['phihkl_'+str(o)],prm['scale']) for o in Orientierungen],axis=0))
	plt.savefig(filename+'_OF.png',dpi=300)
