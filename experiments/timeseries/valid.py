from ctypes import sizeof
from re import X
import pandas as pd
import numpy as np
import copy
import torch
from math import sqrt
import datetime


sliceSum = pd.DataFrame({})
for index in range(21,31):
    data = pd.read_csv('/data/zzw_dataset/TeleMilan/sms-call-internet-mi/sms-call-internet-mi-2013-11-'+str(index).zfill(2) +'.csv')
    data1 = data.fillna(0) # ⽤0替换DataFrame对象中所有的空值
    sliceSum2 = data1.groupby(['time', 'cellid'], as_index=False).sum()
    sliceSum = sliceSum.append(sliceSum2)
    #sliceSum = pd.concat(sliceSum2)
#print(sliceSum)
arr = np.array(sliceSum)
start = copy.deepcopy(pd.Timestamp(arr[0,0]).timestamp())
for i in range(0,len(arr)):
    arr[i,0] = (pd.Timestamp(arr[i,0]).timestamp()- start)/600
data_smsin = np.zeros((100,100,int(arr[-1,0]+1)))
data_smsout = np.zeros((100,100,int(arr[-1,0]+1)))
data_callin = np.zeros((100,100,int(arr[-1,0]+1)))
data_callout = np.zeros((100,100,int(arr[-1,0]+1)))
data_net = np.zeros((100,100,int(arr[-1,0]+1)))
maesi = 0
maeso = 0
maeci =0
maeco = 0
maen = 0
num = 0
for t in range(0,len(arr)):
    i=int(((arr[t,1]-1)//100))
    j=int((arr[t,1]-1)-i*100)
    k=int(arr[t,0])
    data_smsin[i,j,k]=arr[t,4]
    data_smsout[i,j,k]=arr[t,5]
    data_callin[i,j,k]=arr[t,6]
    data_callout[i,j,k]=arr[t,7]
    data_net[i,j,k]=arr[t,8]
    if k>12 and i>29 and i<60 and j>39 and j<70:
        num = num + 1
        maesi = maesi+ abs(data_smsin[i,j,k] - data_smsin[i,j,k-1])
        maeso = maeso+ abs(data_smsout[i,j,k] - data_smsout[i,j,k-1])
        maeci = maeci+ abs(data_callin[i,j,k] - data_callin[i,j,k-1])
        maeco = maeco+ abs(data_callout[i,j,k] - data_callout[i,j,k-1])
        maen = maen+ abs(data_net[i,j,k] - data_net[i,j,k-1])

#print(arr[-1,0])
#num = 10000*(data_sms.shape[2])
print([maesi/num, maeso/num, maeci/num, maeco/num, maen/num, num])