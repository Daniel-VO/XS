import glob
import fabio
import numpy as np
import matplotlib.pyplot as plt

plt.plot([np.sum(fabio.open(i).data) for i in sorted(glob.glob('*[!alle].img'))])
plt.ylim([0,None])
plt.savefig('SC_ints.png')
