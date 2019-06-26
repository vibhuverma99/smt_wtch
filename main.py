import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xlrd
import time

loc = "/home/vibhu/smt_wtch/Raw_PPG_OSRAM_Green_WBottom_3min.xls" #location of excel file

df = pd.read_excel(loc)     
a = df.as_matrix()


x = np.linspace(0,60,num =(60*50))  #making the x time axis each second has 50 divisions
y = a[0:3000 , 1]                   #making y as the array of magnitudes from excel
plt.plot(x,y)
plt.show()
