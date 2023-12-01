"""
Created 01. December 2023 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import glob
import scipy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import crystals
from crystals import Atom,Crystal

filename,q,qx,qy,qz,yobs,sig=[np.load('reflections.npy')[:,i] for i in np.arange(7)]

q0,yobs0,sig0=np.array([float(i) for i in q]),np.array([float(i) for i in yobs]),np.array([float(i) for i in sig])
qcoords0=np.array([np.array([float(i) for i in qx]),np.array([float(i) for i in qy]),np.array([float(i) for i in qz])]).transpose()

checkstring,q,qcoords,yobs,sig=str(),np.array([]),np.array([]),np.array([]),np.array([])

for c,coord in enumerate(qcoords0):
	if filename[c] not in checkstring:
		similar=np.where([scipy.spatial.distance.cosine(coord,qc)<1e-3 for qc in qcoords0])
		checkstring+=str(filename[similar])
		q=np.append(q,np.average(q0[similar]))
		qcoords=np.append(qcoords,np.average(qcoords0[similar],axis=0,weights=yobs0[similar]))
		yobs=np.append(yobs,np.max(yobs0[similar]))
		sig=np.append(sig,np.min(sig0[similar]))
qcoords=qcoords.reshape(-1,3)

args=np.where(yobs>1e5)
plt.errorbar(q,yobs,marker='s',markersize=2,elinewidth=1,capthick=1,capsize=3,linewidth=0)
plt.errorbar(q[args],yobs[args],xerr=sig[args],marker='s',markersize=2,elinewidth=1,capthick=1,capsize=3,linewidth=0)
plt.show()

filename,qcoords,yobs=filename[args],qcoords[args],yobs[args]

plt.close('all')
ax=plt.figure().add_subplot(projection='3d')
ax.scatter(qcoords[:,0],qcoords[:,1],qcoords[:,2],edgecolors='k',c=np.log(yobs),cmap='coolwarm')
ax.set_xlabel('$X$');ax.set_ylabel('$Y$');ax.set_zlabel('$Z$')
ax.set_zlim([0,None])
plt.draw()

# ~ unitcell=[Atom('Si',coords = [0,0,0])];lattice_vectors=5.43070*np.eye(3)
# ~ Crystal(unitcell,lattice_vectors)
# ~ Crystal.from_cif('Si.cif')
# ~ Crystal.from_database('Si')
# ~ lattice=crystals.index_dirax(qcoords,initial=None)	#,length_bounds=(1.95,4)

# ~ for l,valuehkl in enumerate(lattice[1]):
	# ~ ax.text(qcoords[:,0][l],qcoords[:,1][l],qcoords[:,2][l],str([hkl.round(1) for hkl in valuehkl])+' ',ha='right')
	# ~ ax.text(qcoords[:,0][l],qcoords[:,1][l],qcoords[:,2][l],str(' '+filename[l]))

# ~ X=np.linspace(-max(abs(q)),max(abs(q)));Y=np.linspace(-max(abs(q)),max(abs(q)))
# ~ X,Y=np.meshgrid(X,Y)
# ~ for r in lattice[0].scattering_vector([[1,1,1],[2,2,0],[3,1,1]]):
	# ~ Z=(np.linalg.norm(r)**2-X**2-Y**2)**0.5
	# ~ ax.plot_surface(X,Y,Z,color='k',alpha=0.2)

# ~ print(lattice,file=open('lattice.txt','w'))

plt.show()
