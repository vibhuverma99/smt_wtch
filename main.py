import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xlrd
import time
import heartpy as hp
import csv

#variables
freq = 50 #hertz
data = hp.get_data('raw.csv')
time = 120 #in seconds
#working_data,measures = hp.process(data,freq)

#print(measures['bpm'],end=" ")
print(np.shape(data))


x = np.linspace(0,time,num =(time*freq))  #making the x time axis each second has 50 divisions
y = data[2167:(2167 + (time*freq)):1]   #making y as the array of magnitudes from excel
plt.plot(x,y)
plt.show()
