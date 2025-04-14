"""
Created 14. April 2025 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=np.load(str(sys.argv[1])+'.npy',allow_pickle=True)

fnames=data[:,0]
names=[]
values=[]
stderrs=[]
for result in data[:,1]:
	names.append([param.name for param in result.params.values()])
	values.append([param.value for param in result.params.values()])
	stderrs.append([param.stderr for param in result.params.values()])
names=np.array(names).transpose()[:,0]
values=np.array(values).transpose()
stderrs=np.array(stderrs).transpose()

for i,valuei in enumerate(values):
	plt.close('all')
	argsort=np.argsort(fnames)
	plt.errorbar(fnames[argsort],values[i][argsort])#,yerr=stderrs[i])
	plt.setp(plt.gca().xaxis.get_majorticklabels(),rotation=45,ha='right',rotation_mode='anchor')
	plt.tight_layout(pad=0.1)
	plt.savefig(str(sys.argv[1])+'_'+names[i]+'.png')

plt.close('all')
fig,ax1=plt.subplots()
dataframe=pd.DataFrame(np.transpose(values),columns=names)
corr=dataframe.corr()
cax=ax1.matshow(corr,cmap='coolwarm',vmin=-1,vmax=1,interpolation='none')
cbar=fig.colorbar(cax)
ticks=np.arange(0,len(dataframe.columns),1)
ax1.set_xticks(ticks)
ax1.set_yticks(ticks)
ax1.set_xticklabels(dataframe.columns)
ax1.set_yticklabels(dataframe.columns)
plt.xticks(rotation=90)
ax1.set_ylim([len(dataframe.columns)-0.5,-0.5])
plt.tight_layout(pad=0.1)
plt.savefig(str(sys.argv[1])+'_corr.png')
