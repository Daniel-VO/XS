"""
Created 09. Dezember 2025 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import glob
import scipy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def sortedfiles(pattern):
	files0=np.array(glob.glob(pattern),dtype=None)
	files1=np.array(glob.glob(pattern),dtype=None)
	for i,s in enumerate(files1):
		if len(s)<32:
			files1[i]=s.replace('SAXS_','SAXS_00')
		elif len(s)<33:
			files1[i]=s.replace('SAXS_','SAXS_0')
	return files0[np.argsort(files1)]#[::-1]

ind_collect=[]
x_collect=[]
y_collect=[]

for f in sortedfiles('*.dat'):
	print(f)
	filename=os.path.splitext(f)[0]
	ind_collect.append(int(filename.split('_')[-3]))
	x,y=np.genfromtxt(f,unpack=True)
	x_collect.append(x);y_collect.append(y)

plt.close('all')
mpl.rc('text',usetex=True)
mpl.rc('text.latex',preamble=r'\usepackage[helvet]{sfmath}')
plt.figure(figsize=(7.5/2.54,5.3/2.54))

for i,ind in enumerate(ind_collect):
	plt.scatter(x_collect[i],np.full(x_collect[i].shape,ind),c=np.log(y_collect[i]),linewidth=0,marker='s',s=1.7,cmap='coolwarm',
	vmin=3,
	vmax=np.log(max([y for ys in y_collect for y in ys]))/2
	)

plt.xlim([0.005,0.25])
plt.ylim([0,100])

plt.tight_layout(pad=0.1)
plt.savefig('Wasserfall.png',dpi=600)
