"""
Created 01. March 2023 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import glob
import scipy
import numpy as np
import matplotlib.pyplot as plt

for f in glob.glob('*_SAXS*.dat'):
	filename=f.split('_SAXS')[0]
	plt.close('all')

	qS,yobsS=np.genfromtxt(f,unpack=True)
	yobsS=scipy.signal.savgol_filter(yobsS,11,1)

	qS,yobsS=qS[np.argmax(yobsS)+7:],yobsS[np.argmax(yobsS)+7:]		####

	plt.plot(qS,yobsS)
	expq=qS;expyobs=yobsS

	if os.path.isfile(f.replace('_SAXS','_USAXS')):
		qU,yobsU=np.genfromtxt(f.replace('_SAXS','_USAXS'),unpack=True)
		yobsU=scipy.signal.savgol_filter(yobsU,11,1)

		qU,yobsU=qU[np.argmax(yobsU):],yobsU[np.argmax(yobsU):]		####

		args=np.where(qU<qS[0])
		qU,yobsU=qU[args],yobsU[args]

		yobsU*=yobsS[0]/yobsU[-1]

		plt.plot(qU,yobsU)
		expq=np.append(expq,qU);expyobs=np.append(expyobs,yobsU)

	if os.path.isfile(f.replace('_SAXS','_TXRD')):
		qW,yobsW=np.genfromtxt(f.replace('_SAXS','_TXRD'),unpack=True)
		yobsW=scipy.signal.savgol_filter(yobsW,11,1)

		args=np.where(qW>qS[-1])
		qW,yobsW=qW[args],yobsW[args]

		yobsW*=yobsS[-1]/yobsW[0]

		plt.plot(qW,yobsW)
		expq=np.append(expq,qW);expyobs=np.append(expyobs,yobsW)

	plt.xscale('log');plt.yscale('log')
	plt.savefig(filename+'_stitch.png')

	np.savetxt(filename+'_stitch.dat',np.transpose([expq,expyobs]),fmt='%.16f')
