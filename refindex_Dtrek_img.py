"""
Created 21. November 2023 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import numpy as np
import os
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import crystals

reflections=[]

filename,q,qx,qy,qz,yobs,sig=[np.load('reflections.npy')[:,i] for i in np.arange(7)]


print(filename,q,qx,qy,qz,yobs,sig)


# ~ lattice=crystals.index_dirax(reflections)
