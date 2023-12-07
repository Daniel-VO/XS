"""
Created 07. December 2023 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import scipy
import numpy as np
import matplotlib.pyplot as plt

data=np.load('data.npz')
chi,phi,twotheta,yobs=data['chi'],data['phi'],data['twotheta'],data['yobs']

hist,bin_edges=np.histogram(twotheta,bins=len(np.unique(twotheta)),weights=yobs)
np.savetxt('ttints.xy',np.array([bin_edges[1:],hist]).transpose(),fmt='%.6f')

chi,phi,twotheta=np.radians([chi,phi,twotheta])
q=4*np.pi*np.sin(twotheta/2)/1.5406
qx= q*np.sin(chi)*np.sin(phi)
qy=-q*np.sin(chi)*np.cos(phi)
qz= q*np.cos(chi)

limit=np.where((yobs*q**2>1000))
q0,qx0,qy0,qz0,yobs0=q[limit],qx[limit],qy[limit],qz[limit],yobs[limit]

coords=np.array([qx0,qy0,qz0]).transpose()
distances=scipy.spatial.distance.cdist(coords,coords,'euclidean')

indices,q,qx,qy,qz,yobs,sigq=np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])
for i,valuei in enumerate(distances):
	if i not in indices:
		similar=np.where(valuei<1)
		indices=np.append(indices,similar)
		q=np.append(q,np.average(q0[similar],weights=yobs0[similar]**2))
		qx=np.append(qx,np.average(qx0[similar],weights=yobs0[similar]**2))
		qy=np.append(qy,np.average(qy0[similar],weights=yobs0[similar]**2))
		qz=np.append(qz,np.average(qz0[similar],weights=yobs0[similar]**2))
		yobs=np.append(yobs,np.max(yobs0[similar]))
		sigq=np.append(sigq,np.std(q0[similar]))

plt.close('all')
ax=plt.figure().add_subplot(projection='3d')
ax.scatter(qx0,qy0,qz0,edgecolors=None,c=yobs0*q0**2,cmap='coolwarm',s=1)
ax.scatter(qx,qy,qz,edgecolors='k',c=yobs*q**2,cmap='coolwarm')
ax.set_xlabel('$X$');ax.set_ylabel('$Y$');ax.set_zlabel('$Z$')
ax.set_zlim([0,None])
plt.show()

np.save('reflections.npy',[q,qx,qy,qz,yobs,sigq])

os.system('python3 refindex.py')
