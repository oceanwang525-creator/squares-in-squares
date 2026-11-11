import sys; sys.path.insert(0,'poc')
from colmodel import *
L=20.535534; h=L/2
for fl in [(False,False),(True,True),(False,True),(True,False)]:
    P=[block_poly(3+h,-5.99+h,5,5,fl[0]),block_poly(-6+h,3+h,5,5,fl[1])]
    print(fl,best(P,L,50,0.01),best(P,L,50,0.01,True))
print('none',best([],L,0,0.05))
