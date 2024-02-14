import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

for f in glob.glob('*_USAXS*.dat'):
	filename=f.split('_USAXS')[0]

	if os.path.isfile(f.replace('_USAXS','_SAXS')):
		qU,yobsU,epsU=np.genfromtxt(f,unpack=True)
		qS,yobsS,epsS=np.genfromtxt(f.replace('_USAXS','_SAXS'),unpack=True)

		overlapU=np.where(qU>=qS[0]);overlapS=np.where(qS<=qU[-1])
		intU=interpolate.interp1d(qU[overlapU],yobsU[overlapU])
		yobsS*=np.average(intU(qS[overlapS][1:])/yobsS[overlapS][1:],weights=yobsS[overlapS][1:])

		plt.close('all')
		plt.plot(qU,yobsU)
		plt.plot(qS,yobsS)
		plt.xscale('log');plt.yscale('log')
		plt.savefig(filename+'_USAXS_SAXS.png')

		args=np.where(qU<=qS[0])
		qU,yobsU,epsU=qU[args],yobsU[args],epsU[args]

		np.savetxt(filename+'_USAXS_SAXS.dat',np.transpose([np.append(qU,qS),np.append(yobsU,yobsS),np.append(epsU,epsS)]),fmt='%.16f')
