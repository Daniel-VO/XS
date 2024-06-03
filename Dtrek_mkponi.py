"""
Created 31. May 2024 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import os
import sys
import fabio
import numpy as np

def take(img,headerkey,indices):
	return np.fromstring(img.header[headerkey],sep=' ')[indices]

img=fabio.open(sys.argv[1])
filename=os.path.splitext(sys.argv[1])[0]

# ~ print(img.header,file=open('header','w'));a=b
detdist=take(img,'PXD_GONIO_VALUES',-1);beamcenterX,beamcenterY,pxsizeX,pxsizeY=take(img,'PXD_SPATIAL_DISTORTION_INFO',[0,1,2,3]);wavelength=take(img,'SOURCE_WAVELENGTH',-1);omega,chi,phi=take(img,'CRYSTAL_GONIO_VALUES',[0,1,2])

print(
'Detector: detector\n'
'Detector_config: {"pixel1": '+str(pxsizeX/1e3)+', "pixel2": '+str(pxsizeY/1e3)+', "max_shape": ['+str(img.data.shape[0])+', '+str(img.data.shape[1])+']}\n'
'Distance: '+str(detdist/1e3)+'\n'
'Poni1: '+str(beamcenterY*pxsizeX/1e3)+'\n'
'Poni2: '+str(beamcenterX*pxsizeX/1e3)+'\n'
'Rot1: 0.0\n'
'Rot2: 0.0\n'
'Rot3: 0.0\n'
'Wavelength: '+str(wavelength/1e10)
,file=open(filename+'.poni','w'))

