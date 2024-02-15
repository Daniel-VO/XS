"""
Created 14. February 2023 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

for f in glob.glob('*_SAXS*.dat'):
	filename=f.split('_SAXS')[0]
	plt.close('all')

	qS,yobsS,epsS=np.genfromtxt(f,unpack=True)
	plt.plot(qS,yobsS)

	if os.path.isfile(f.replace('_SAXS','_USAXS')):
		qU,yobsU,epsU=np.genfromtxt(f.replace('_SAXS','_USAXS'),unpack=True)

		overlapU=np.where(qU>=qS[0]);overlapS=np.where(qS<=qU[-1])
		intU=interpolate.interp1d(qU[overlapU],yobsU[overlapU])
		yobsU*=np.average(yobsS[overlapS][1:]/intU(qS[overlapS][1:]),weights=yobsS[overlapS][1:])

		args=np.where(qU<=qS[0])
		qU,yobsU,epsU=qU[args],yobsU[args],epsU[args]

		plt.plot(qU,yobsU)

	if os.path.isfile(f.replace('_SAXS','_TXRD')):
		qW,yobsW,epsW=np.genfromtxt(f.replace('_SAXS','_TXRD'),unpack=True)

		args=np.where(qW>=qS[-1])
		qW,yobsW,epsW=qW[args],yobsW[args],epsW[args]

		yobsW*=yobsS[-1]/yobsW[0]

		plt.plot(qW,yobsW)

	plt.xscale('log');plt.yscale('log')
	plt.savefig(filename+'_stitch.png')

	# ~ np.savetxt(filename+'_stitch.dat',np.transpose([np.append(qU,qS,qW),np.append(yobsU,yobsS,yobsW),np.append(epsU,epsS,epsW)]),fmt='%.16f')
