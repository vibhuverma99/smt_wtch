import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xlrd
import time
import heartpy as hp
import csv
import scipy.fftpack                 
from scipy.signal import butter , lfilter
import adaptivefilter as af

def butter_bandpass(lowcut, highcut , fs , order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y



if __name__=='__main__':

    #variables
    freq = 50 #hertz
    data = hp.get_data('raw.csv')
    ref = hp.get_data('ceeri.csv')
    end_time = 360 #seconds
    start_time = 0 #seconds
    time_interval = end_time - start_time
    start_offset = start_time * freq
    num =(time_interval*freq)
    #working_data,measures = hp.process(data,freq)
    #print(measures['bpm'],end=" ")
    ##########  print(np.shape(data))


    x = np.linspace(start_time,end_time,num)  #making the x time axis each second has 50 divisions
    y = data[2167+ start_offset:(2167 + (time_interval*freq)+start_offset):1]   #making y as the array of magnitudes from excel
    #somno 6914 freq 128  ceeri 2167 50 hertz
    yref = ref[2167 + start_offset:(2167+start_offset+(time_interval*freq)):1]
    plt.subplot(2,2,1)
    plt.plot(x,y)
    plt.xlabel('\n Time(s)')
    plt.ylabel('Amplitude \n ')


    yf = scipy.fftpack.fft(y)
    freqs = scipy.fftpack.fftfreq(len(y)) * freq
    plt.subplot(2,2,2)
    print('THe size of freqs is',np.shape(freqs))
    print('THe size of yf is',np.shape(yf))
    plt.plot(freqs[1:],yf[1:])
    
    #apply butterworth
    yn = butter_bandpass_filter(y,0.5,4,freq,order=3)
    plt.subplot(2,2,3)
    plt.plot(x,yn)
    plt.xlabel('\n Time(s)')
    plt.ylabel('Amplitude \n ')
    

    plt.subplot(2,2,4)
    plt.plot(x,yref)
    plt.xlabel('\n Time(s)')
    plt.ylabel('Amplitude \n ')
    plt.show()


    
    err = yref - yn
    perr = np.zeros(len(err))

    for i in range(len(yref)):
        perr[i] = (err[i]/yref[i])*100
        print("Error = ",perr[i],'%')
    
    finerr = np.mean(perr)
    print('Overall ERROR is percentage ',finerr)
   

    x = np.linspace(-40 , 40 , 10000)
    n1 = np.random.normal(0,1,10000)
    sig = np.sin(x)+n1
    n2 = np.random.normal(0,1,10000)
    M = 9 #desired no. of filter taps i.e length of filter
    #initialCoeff = np.random.randint(0,10,M)
    initialCoeff = np.zeros(M)
    freq = 10000/80

    #filtering
    sigf = butter_bandpass_filter(sig,0.01,4,freq,order=3)


    y,e,w = af.lms(n2,sigf,M,0.01,0,initialCoeff,9000,True)
    
    plt.subplot(3,2,1)
    plt.plot(x,np.sin(x))
    plt.xlabel('TIME')
    plt.ylabel('AMP')
    plt.title('SIGNAL WITHOUT NOISE')

    plt.subplot(3,
    2,5)
    plt.plot(x,sig)
    plt.xlabel('TIME')
    plt.ylabel('AMP')
    plt.title('SIGNAL WITH ADDED NOISE')

    plt.subplot(3,2,3)
    plt.plot(x,n1)
    plt.xlabel('TIME')
    plt.ylabel('AMP')
    plt.title('NOISE N1')

    plt.subplot(3,2,4)
    plt.plot(x,n2)
    plt.xlabel('TIME')
    plt.ylabel('AMP')
    plt.title('NOISE N2')

    plt.subplot(3,2,6)
    plt.plot(x[0:len(y):1],e)
    plt.xlabel('TIME')
    plt.ylabel('AMP')
    plt.title('SIGNAL FILTERED')

    plt.subplot(3,2,2)
    plt.plot(x[0:len(y):1],y)
    plt.xlabel('TIME')
    plt.ylabel('AMP')
    plt.title('SIGNAL Y')

    plt.show()
