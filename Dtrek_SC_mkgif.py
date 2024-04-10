import fabio
import glob
import numpy as np
import PIL
import matplotlib as mpl

frames=[]

for i in sorted(glob.glob('*[!alle].img')):
	frames.append(PIL.Image.fromarray(np.uint8(mpl.cm.afmhot(fabio.open(i).data)*255)))

frames[0].save('animation.gif',format='GIF',append_images=frames,save_all=True,optimize=True,disposal=2,duration=150,loop=0)
