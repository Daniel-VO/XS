"""
Created 03. Mai 2024 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import ray
import glob
import numpy as np

@ray.remote
def collect(f):
	chi_collect,twotheta_collect,yobs_collect=[],[],[]
	for line in open(f,'r').readlines():
		if line.split()[0]=='*MEAS_COND_AXIS_POSITION-2':
			chi=float(line.split()[1].replace('"',''))
		if line.split()[0]=='*MEAS_COND_AXIS_POSITION-5':
			phi=float(line.split()[1].replace('"',''))
		if '*' not in line:
			chi_collect.append(chi)
			twotheta_collect.append(float(line.split()[0]))
			yobs_collect.append(float(line.split()[1]))
	print(phi)
	yobs,chi,twotheta=np.histogram2d(chi_collect,twotheta_collect,weights=yobs_collect,\
					  bins=[np.arange(min(chi_collect),max(chi_collect)),\
							np.arange(min(twotheta_collect),max(twotheta_collect),step=0.1)])
	chi,twotheta=np.meshgrid(chi[1:],twotheta[1:]);phi=np.full(yobs.shape,phi);yobs=yobs.transpose()
	return chi.flatten(),phi.flatten(),twotheta.flatten(),yobs.flatten()

chi,phi,twotheta,yobs=np.concatenate(ray.get([collect.remote(f) for f in glob.glob('*.ras')]),axis=1)
np.save('RSS.npy',[chi,phi,twotheta,yobs])

yobs,twotheta=np.histogram(twotheta,weights=yobs/np.sin(np.radians(twotheta)/2),bins=np.unique(twotheta))
np.savetxt('ttints.xy',np.array([twotheta[1:],yobs]).transpose(),fmt='%.6f')

os.system('python3 2D-RSS_refsearch.py')
