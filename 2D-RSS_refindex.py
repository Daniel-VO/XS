"""
Created 03. Mai 2024 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import numpy as np
import matplotlib.pyplot as plt
import crystals
from crystals import Atom,Crystal

q,qx,qy,qz,yobs,sigq=np.load('reflections.npy')

r=scipy.spatial.transform.Rotation.from_euler('z',45,degrees=True)
for i,valuei in enumerate(q):
	qx[i],qy[i],qz[i]=r.apply([qx[i],qy[i],qz[i]])

guess=crystals.lattice.Lattice.from_parameters(a=5.43070,b=5.43070,c=5.43070,alpha=90.00,beta=90.00,gamma=90.00)
lattice=crystals.index_dirax(np.array([qx,qy,qz]).transpose(),initial=guess,length_bounds=(2,3))

plt.close('all')
ax=plt.figure().add_subplot(projection='3d')
ax.scatter(qx,qy,qz,edgecolors='k',c=yobs,cmap='coolwarm')
ax.set_xlabel('$X$');ax.set_ylabel('$Y$');ax.set_zlabel('$Z$')
ax.set_zlim([0,None])

for l,valuehkl in enumerate(lattice[1]):
	ax.text(qx[l],qy[l],qz[l],str([hkl.round(1) for hkl in valuehkl]).replace('[','(').replace(']',')')+' ')

# ~ X=np.linspace(-max(abs(q)),max(abs(q)));Y=np.linspace(-max(abs(q)),max(abs(q)))
# ~ X,Y=np.meshgrid(X,Y)
# ~ for r in lattice[0].scattering_vector([[1,0,0]]):
	# ~ Z=(np.linalg.norm(r)**2-X**2-Y**2)**0.5
	# ~ ax.plot_surface(X,Y,Z,color='k',alpha=0.2)

print(lattice,file=open('lattice.txt','w'))

plt.show()
