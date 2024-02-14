"""
Created 14. February 2023 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

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
		ovU=interpolate.interp1d(qU[overlapU],yobsU[overlapU])
		yobsS*=np.average(ovU(qS[overlapS][1:])/yobsS[overlapS][1:],weights=yobsS[overlapS][1:])

		plt.close('all')
		plt.plot(qU,yobsU)
		plt.plot(qS,yobsS)
		plt.xscale('log');plt.yscale('log')
		plt.savefig(filename+'_USAXS_SAXS.png')

		args=np.where(qU<=qS[0])
		qU,yobsU,epsU=qU[args],yobsU[args],epsU[args]

		np.savetxt(filename+'_USAXS_SAXS.dat',np.transpose([np.append(qU,qS),np.append(yobsU,yobsS),np.append(epsU,epsS)]),fmt='%.8f')
