"""
Created 20. March 2024 by Daniel Van Opdenbosch, Technical University of Munich

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
		[1e-3,4e-2],
		[4e-2,7e-1],
		[7e-1,1e1]
		]

os.system('mv results.log results.alt')
for r in ranges:
	for f in glob.glob('*_stitch.dat'):
		filename=os.path.splitext(f)[0]+'_'+str(r)
		q,yobs=np.genfromtxt(f,unpack=True)

		q,yobs=q[np.argmax(yobs):],yobs[np.argmax(yobs):]						####

		plt.close('all')
		plt.plot(q,yobs)

		args=np.where((q>=r[0])&(q<=r[1]))										####

		profile=raw.make_profile(q[args],yobs[args],np.ones(len(args[0])),filename)
		guinier=raw.auto_guinier(profile)
		ift=raw.bift(profile)

		plt.plot(ift[0].q_orig,ift[0].i_orig)
		plt.plot(ift[0].q_orig,ift[0].i_fit)
		plt.xscale('log');plt.yscale('log');plt.xlim([1e-4,None])
		plt.savefig(filename+'_iq.png')

		plt.close('all')
		plt.plot(ift[0].r,ift[0].p)
		plt.savefig(filename+'_pr.png')

		raw.save_ift(ift[0],filename+'.ift')
		raw.save_report(filename+'.pdf',profiles=[profile],ifts=[ift[0]])
		print(filename,'rg =',guinier[0],'+-',guinier[2],' bift_rg =',ift[2],'+-',ift[5],file=open('results.log','a'))
