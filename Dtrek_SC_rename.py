import os
import glob

os.system('mkdir CB_BG')

for f in glob.glob('*.img'):
	filename=os.path.splitext(f)[0]
	if 'CB' in f or 'BG' in f:
		print(f)
		os.system('mv '+f+' '+os.getcwd()+'/CB_BG')
	else:
		if len(filename.split('_')[-1])<3:
			while len(filename.split('_')[-1])<3:
				filename=filename[:-len(filename.split('_')[-1])]+'0'+filename[-len(filename.split('_')[-1]):]
			os.system('mv '+f+' '+filename+'.img')
