"""
Created 30. November 2023 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import crystals
from crystals import Atom,Crystal

filename,q,qx,qy,qz,yobs,sig=[np.load('reflections.npy')[:,i] for i in np.arange(7)]

q=np.array([float(i) for i in q])
qx=np.array([float(i) for i in qx])
qy=np.array([float(i) for i in qy])
qz=np.array([float(i) for i in qz])
yobs=np.array([float(i) for i in yobs])
sig=np.array([float(i) for i in sig])

args=np.where(yobs*q**2>1e6)

# ~ plt.errorbar(q,yobs*q**2,marker='s',markersize=2,elinewidth=1,capthick=1,capsize=3,linewidth=0)
# ~ plt.errorbar(q[args],yobs[args]*q[args]**2,xerr=sig[args],marker='s',markersize=2,elinewidth=1,capthick=1,capsize=3,linewidth=0)
# ~ plt.show()

filename,qx,qy,qz,yobs=filename[args],qx[args],qy[args],qz[args],yobs[args]

plt.close('all')
ax=plt.figure().add_subplot(projection='3d')
ax.scatter(qx,qy,qz,edgecolors='k',c=np.log(yobs),cmap='coolwarm')
ax.set_xlabel('$X$');ax.set_ylabel('$Y$');ax.set_zlabel('$Z$')
ax.set_zlim([0,None])
plt.draw()

# ~ unitcell=[Atom('Si',coords = [0,0,0])];lattice_vectors=5.43070*np.eye(3)
# ~ Crystal(unitcell,lattice_vectors)
# ~ Crystal.from_cif('Si.cif')
# ~ Crystal.from_database('Si')
lattice=crystals.index_dirax(np.array([qx,qy,qz]).transpose(),initial=None,length_bounds=(1.95,4))

# ~ for l,valuehkl in enumerate(lattice[1]):
	# ~ ax.text(qx[l],qy[l],qz[l],str([hkl.round(1) for hkl in valuehkl])+' ',ha='right')
	# ~ ax.text(qx[l],qy[l],qz[l],str(' '+filename[l]))

X=np.linspace(-max(abs(q)),max(abs(q)));Y=np.linspace(-max(abs(q)),max(abs(q)))
X,Y=np.meshgrid(X,Y)
for r in lattice[0].scattering_vector([[1,1,1],[2,2,0],[3,1,1]]):
	Z=(np.linalg.norm(r)**2-X**2-Y**2)**0.5
	ax.plot_surface(X,Y,Z,color='k',alpha=0.2)

print(lattice,file=open('lattice.txt','w'))

plt.show()

