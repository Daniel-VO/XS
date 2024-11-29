"""
Created 29. November 2024 by Daniel Van Opdenbosch, Technical University of Munich

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
		[7e-3,1e0],
		]

os.system('mv results.log results.alt')
for r in ranges:
	for f in glob.glob('**/*.dat'):
		filename=os.path.splitext(f)[0]+'_'+str(r)
		if not os.path.exists(filename+'.pdf'):
			print(filename)
			q,yobs=np.genfromtxt(f,unpack=True)

			args=np.where((q>=r[0])&(q<=r[1]))									####
			q,yobs=q[args],yobs[args]											####
			q,yobs=q[np.argmax(yobs):],yobs[np.argmax(yobs):]					####

			args=np.where((q>=r[0])&(q<=r[1]))									####
			profile=raw.make_profile(q[args],yobs[args],np.ones(len(q[args])),filename)
			ift=raw.bift(profile)

			profile=raw.make_profile(q,yobs,np.ones(len(q)),filename)
			guinier=raw.auto_guinier(profile)

			plt.close('all')
			plt.plot(q,yobs)
			plt.plot(ift[0].q_orig,ift[0].i_orig)
			plt.plot(ift[0].q_orig,ift[0].i_fit)
			plt.xscale('log');plt.yscale('log');plt.xlim([r[0]/2,None])
			plt.savefig(filename+'_iq.png')

			plt.close('all')
			plt.plot(ift[0].r,ift[0].p)
			plt.savefig(filename+'_pr.png')

			raw.save_ift(ift[0],filename+'.ift')
			raw.save_report(filename+'.pdf',profiles=[profile],ifts=[ift[0]])
			print(filename,'rg =',guinier[0],'+-',guinier[2],' bift_rg =',ift[2],'+-',ift[5],file=open('results.log','a'))
