"""
Created 20. Mai 2026 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import sys
import glob
import scipy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sas.sascalc.pr.invertor import Invertor

def line_prepender(filename, line):
	with open(filename, 'r+') as f:
		content = f.read()
		f.seek(0, 0)
		f.write(line.rstrip('\r\n') + '\n' + content)

ranges=[
		[7e-3,3e-1],
		]

os.system('mv results.log results.alt')
for ran in ranges:
	for f in glob.glob('*.dat'):
		filename=os.path.splitext(f)[0]+'_'+str(ran)
		print(filename)
		q,yobs=np.genfromtxt(f,unpack=True)
		slinf=eval(open(f).readlines()[0].split('#')[-1])
		invertor=Invertor()

		args=(q>=ran[0])&(q<=ran[1])											####
		q,yobs=q[args],yobs[args]												####
		q,yobs=q[np.argmax(yobs):],yobs[np.argmax(yobs):]						####
		invertor.set_dmax(2*np.pi/min(q))										####
		invertor.alpha=0														####
		invertor.nfunc=20														####

		invertor.set_x(q); invertor.set_y(yobs); invertor.set_err(np.ones(len(q)))
		if 'dIW' in slinf.keys():
			invertor.set_slit_height(slinf['dIW']); invertor.set_slit_width(slinf['bxw'])
		else:
			invertor.set_slit_height(slinf['bxw']); invertor.set_slit_width(slinf['bxw'])
		out,cov=invertor.invert(nfunc=invertor.nfunc)
		r=np.linspace(0,invertor.d_max,num=len(q))
		pr=invertor.pr_err(out,cov,r)

		plt.close('all')

		plt.plot(q,yobs)
		plt.plot(q,invertor.get_iq_smeared(out,q))
		plt.xscale('log');plt.yscale('log')
		plt.savefig(filename+'_iq.png')

		plt.close('all')
		plt.plot(r,pr[0])
		plt.fill_between(r,pr[0]+pr[1]/2,pr[0]-pr[1]/2,alpha=0.5)
		plt.savefig(filename+'_pr.png')

		invertor.to_file(filename+'_pr.txt')
		line_prepender(filename+'_pr.txt','#iq0='+str('%.8e'%invertor.iq0(out)))
		line_prepender(filename+'_pr.txt','#rg='+str('%.4f'%invertor.rg(out)))
