# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 10:01:17 2021

@author: czw
"""
import matplotlib.pyplot as plt
import numpy as np
pe=[]
pacc=[]
pc=[]
import csv
with open('testresult.csv', 'r') as f:
    
    reader = csv.reader(f)
    result=list(reader)
    
for i in range(1,len(result)):
    a=result[i][0].split('[')[-1]
    pe.append(int(a))
    pacc.append(np.float64(result[i][1]))
    b=result[i][2].split(']')[0]
    pc.append(np.float64(b))
plt.figure()
plt.plot(pe,pacc,'r',label="test_acc")
plt.legend(loc="upper left") #显示图中标签
plt.xlabel("the number of epoch")
plt.ylabel("value of acc")
plt.show()
plt.figure()
plt.plot(pe,pc,'b',label="total_cost")
plt.legend(loc="upper right") #显示图中标签
plt.xlabel("the number of epoch")
plt.ylabel("value of cost")
plt.show()
    
