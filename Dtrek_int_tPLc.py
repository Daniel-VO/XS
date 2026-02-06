import os
import glob
import numpy as np

for f in glob.glob('*.xy'):
	if '2th_deg' in open(f,'r').read():
		fn=os.path.splitext(f)[0]
		tt,yobs=np.genfromtxt(f,unpack=True,skip_header=24)
		np.savetxt('Profex/'+fn+'_profex.xy',np.transpose([tt,yobs/np.sin(np.radians(tt))]),fmt='%.8f')
