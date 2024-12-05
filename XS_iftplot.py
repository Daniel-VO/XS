"""
Created 05. Dezember 2024 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import sys
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import bioxtasraw.RAWAPI as raw

ranges=[
		[7e-3,3e-1],
		]

for r in ranges:
	ifts=sorted(raw.load_ifts(glob.glob('*'+str(r)+'*.out')),key=lambda i: [i.getParameter('filename').replace('SAXS_','SAXS_0') if i.getParameter('filename').split('_bgs_toq_')[0][-2]=='_' else i.getParameter('filename')])

	names=np.array([])
	for i in ifts:
		names=np.append(names,i.getParameter('filename').split('_SAXS_')[0])

	for name in np.unique(names):
		plt.close('all')

		for i,w in enumerate(np.where(names==name)[0]):
			plt.plot(ifts[w].r,ifts[w].p,label=ifts[w].getParameter('filename').split('_bgs_toq_')[0]+': '+str(round(ifts[w].getParameter('rg'),2)))
		plt.legend(fontsize=6,frameon=False)

		plt.xlim([None,np.max([ift.r for ift in ifts])*1.02])
		plt.tight_layout(pad=0)
		plt.savefig(name+str(r)+'.png',dpi=300)
