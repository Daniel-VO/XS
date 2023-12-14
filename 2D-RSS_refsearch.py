"""
Created 14. December 2023 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import scipy
import numpy as np
import matplotlib.pyplot as plt

data=np.load('data.npz')
chi,phi,twotheta,yobs=data['chi'],data['phi'],data['twotheta'],data['yobs']
chi,phi,twotheta=np.radians([chi,phi,twotheta])

Pol=1+np.cos(twotheta)**2
Lor=1/np.sin(twotheta)
yobs/=Pol*Lor

q=4*np.pi*np.sin(twotheta/2)/1.5406
qx= q*np.sin(chi)*np.sin(phi)
qy=-q*np.sin(chi)*np.cos(phi)
qz= q*np.cos(chi)

plt.close('all')
plt.scatter(q,yobs,marker='.',s=1)

limit=np.max(yobs)/2*np.exp(-q**2/30)
plt.scatter(q,limit,marker='.',s=1)
cutoff=np.where(yobs>limit)
q0,qx0,qy0,qz0,yobs0=q[cutoff],qx[cutoff],qy[cutoff],qz[cutoff],yobs[cutoff]

coords=np.array([qx0,qy0,qz0]).transpose()
distances=scipy.spatial.distance.cdist(coords,coords,'euclidean')

indices,q,qx,qy,qz,yobs,sigq=np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])
for i,valuei in enumerate(distances):
	if i not in indices:
		similar=np.where(valuei<1)
		indices=np.append(indices,similar)
		q=np.append(q,np.average(q0[similar],weights=yobs0[similar]))
		qx=np.append(qx,np.average(qx0[similar],weights=yobs0[similar]))
		qy=np.append(qy,np.average(qy0[similar],weights=yobs0[similar]))
		qz=np.append(qz,np.average(qz0[similar],weights=yobs0[similar]))
		yobs=np.append(yobs,np.max(yobs0[similar]))
		sigq=np.append(sigq,np.std(q0[similar]))

		plt.scatter(q0[similar],yobs0[similar],marker='.',s=1)
		plt.errorbar(q[-1],yobs[-1],xerr=sigq[-1],marker='s',markersize=1,elinewidth=1,capthick=1,capsize=2,linewidth=0)
plt.savefig('ints.png',dpi=300)

plt.close('all')
ax=plt.figure().add_subplot(projection='3d')
ax.scatter(qx0,qy0,qz0,edgecolors=None,c=yobs0,cmap='coolwarm',s=1)
ax.scatter(qx,qy,qz,edgecolors='k',c=yobs,cmap='coolwarm')
for v,valuev in enumerate(q):
	ax.text(qx[v],qy[v],qz[v],str(valuev.round(2)))
ax.set_xlabel('$X$');ax.set_ylabel('$Y$');ax.set_zlabel('$Z$')
ax.set_zlim([0,None])
plt.show()

np.save('reflections.npy',[q,qx,qy,qz,yobs,sigq])

os.system('python3 2D-RSS_refindex.py')
