"""
Created 16. Mai 2025 by Daniel Van Opdenbosch, Technical University of Munich

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

f=10;s=10;lamb=1.5406;rGon=300;d1=173.5;d2=2*rGon-d1
AHlen=(f/2+s/2)/d1*2*rGon
LHlen=(f/2+s/2)/d1*d2
a=4*np.pi*np.sin(np.arctan(AHlen/2/rGon))/lamb
b=4*np.pi*np.sin(np.arctan(LHlen/2/rGon))/lamb
dIW=4*np.pi*np.sin(np.arctan((AHlen-LHlen)/2/rGon))/lamb

ranges=[
		[7e-3,3e-1],
		]

os.system('mv results.log results.alt')
for ran in ranges:
	for f in glob.glob('*.dat'):
		filename=os.path.splitext(f)[0]+'_'+str(ran)
		print(filename)
		q,yobs=np.genfromtxt(f,unpack=True)

		args=np.where((q>=ran[0])&(q<=ran[1]))									####
		q,yobs=q[args],yobs[args]											####
		q,yobs=q[np.argmax(yobs):],yobs[np.argmax(yobs):]					####

		plt.close('all')

		invertor=Invertor()
		invertor.set_x(q);invertor.set_y(yobs);invertor.set_err(np.ones(len(q)))
		invertor.set_slit_height(dIW);invertor.set_slit_width(float(open(f).readlines()[0].split('=')[-1]))

		out,cov=invertor.invert(nfunc=int(invertor.get_dmax()/11))
		r=np.arange(0.0,invertor.d_max,invertor.d_max/100)
		pr=invertor.pr_err(out,cov,r)

		for _ in np.arange(10):
			print(np.any(pr[0]<0),invertor.get_dmax())
			if np.any(pr[0]<0):
				invertor.set_dmax(1.2*r[np.where(pr[0]<0)][0])
			else:
				invertor.set_dmax(1.2*invertor.get_dmax())

			out,cov=invertor.invert(nfunc=int(invertor.get_dmax()/11))
			r=np.arange(0.0,invertor.d_max,invertor.d_max/100)
			pr=invertor.pr_err(out,cov,r)

		plt.plot(q,yobs)
		plt.plot(q,invertor.get_iq_smeared(out,q))
		plt.xscale('log');plt.yscale('log')
		plt.savefig(filename+'_iq.png')

		plt.close('all')
		plt.errorbar(r,pr[0],yerr=pr[1])
		plt.savefig(filename+'_pr.png')

		invertor.to_file(filename+'_pr.txt')
		line_prepender(filename+'_pr.txt','#iq0='+str('%.8e'%invertor.iq0(out)))
		line_prepender(filename+'_pr.txt','#rg='+str('%.4f'%invertor.rg(out)))
