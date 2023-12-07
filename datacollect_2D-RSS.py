"""
Created 07. December 2023 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import ray
import glob
import numpy as np

@ray.remote
def collect(f):
	chi_collect,phi_collect,twotheta_collect,yobs_collect=[],[],[],[]
	for line in open(f,'r').readlines():
		if line.split()[0]=='*MEAS_COND_AXIS_POSITION-2':
			chi=float(line.split()[1].replace('"',''))
		if line.split()[0]=='*MEAS_COND_AXIS_POSITION-5':
			phi=float(line.split()[1].replace('"',''))
		if '*' not in line:
			chi_collect.append(chi);phi_collect.append(phi)
			twotheta_collect.append(float(line.split()[0]))
			yobs_collect.append(float(line.split()[1]))
	return chi_collect,phi_collect,twotheta_collect,yobs_collect

chi,phi,twotheta,yobs=np.concatenate(ray.get([collect.remote(f) for f in glob.glob('*.ras')]),axis=1)
np.savez_compressed('data.npz',chi=chi,phi=phi,twotheta=twotheta,yobs=yobs)

os.system('python3 refsearch_2D-RSS.py')
