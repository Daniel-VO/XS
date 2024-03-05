"""
Created 04. March 2023 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import glob
import scipy
import numpy as np
import matplotlib.pyplot as plt

smpoints=21

for f in glob.glob('*_SAXS*.dat'):
	filename=f.split('_SAXS')[0]

	qS,yobsS=np.genfromtxt(f,unpack=True)
	yobsS=scipy.signal.savgol_filter(yobsS,smpoints,1)

	if os.path.isfile(f.replace('_SAXS','_USAXS')):
		qU,yobsU=np.genfromtxt(f.replace('_SAXS','_USAXS'),unpack=True)
		yobsU=scipy.signal.savgol_filter(yobsU,smpoints,1)

		if np.any(yobsU<min(yobsS)):
			args=np.where(yobsU<min(yobsS))[0][0]
			qU,yobsU=qU[:args],yobsU[:args]

		if len(qU)>0:
			args=np.where(qS>max(qU))
			qS,yobsS=qS[args],yobsS[args]
			yobsS*=yobsU[-1]/yobsS[0]

	if os.path.isfile(f.replace('_SAXS','_TXRD')):
		qW,yobsW=np.genfromtxt(f.replace('_SAXS','_TXRD'),unpack=True)
		yobsW=scipy.signal.savgol_filter(yobsW,smpoints,1)

		# ~ if np.any(yobsS<min(yobsW)):
			# ~ args=np.where(yobsS<min(yobsW))[0][0]
			# ~ qS,yobsS=qS[:args],yobsS[:args]

		if len(qS)>0:
			args=np.where(qW>max(qS))
			qW,yobsW=qW[args],yobsW[args]
			yobsW*=yobsS[-1]/yobsW[0]

	plt.close('all')

	plt.plot(qU,yobsU)
	plt.plot(qS,yobsS)
	plt.plot(qW,yobsW)

	plt.xscale('log');plt.yscale('log')
	plt.savefig(filename+'_stitch.png')

	np.savetxt(filename+'_stitch.dat',np.transpose([np.concatenate((qU,qS,qW)),np.concatenate((yobsU,yobsS,yobsW))]),fmt='%.16f')
