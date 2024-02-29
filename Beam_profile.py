import numpy as np
import sys

f=10
s=10
print(s)

lamb=1.5406
rGon=300
d1=173.5
d2=2*rGon-d1

AHlen=(f/2+s/2)/d1*2*rGon
LHlen=(f/2+s/2)/d1*d2

AHrez=4*np.pi*np.sin(np.arctan(AHlen/rGon/2))/lamb
LHrez=4*np.pi*np.sin(np.arctan(LHlen/rGon/2))/lamb

print('AHlen: ',AHlen,'mm')
print('LHlen: ',LHlen,'mm')

print('AHrez: q=',AHrez,'A^-1')
print('LHrez: q=',LHrez,'A^-1')

print('AHrez Kontrolle: q=',2*np.pi/lamb*AHlen/rGon,'A^-1')
print('LHrez Kontrolle: q=',2*np.pi/lamb*LHlen/rGon,'A^-1')

print('d2 Kontrolle: ',d2)
print('F Kontrolle: ',2*np.pi/lamb/rGon)

print('AHrez-LHrez Kontrolle: q=',AHrez-LHrez,'A^-1')
print('f/(2rGon) Kontrolle: q=',4*np.pi*np.sin(np.arctan((AHlen-LHlen)/rGon/2))/lamb,'A^-1')
