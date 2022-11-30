# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["image.origin"] = 'lower'
def mag2db(x):
    return 20*np.log10(np.abs(x))

def extentVals(xLims,delX,yLims,delY):
    left = xLims[0] - delX/2
    right = xLims[1] + delX/2
    bottom = yLims[0] - delY/2
    top = yLims[1] + delY/2

    return (left,right,bottom,top)

fs = 1e6
fc = 0.1*fs
nSampsPerFrame = 1024
nFrames = 400
nSamps = nSampsPerFrame * nFrames
nFFT = 1024

t = np.arange(nSamps)/fs
x = np.sin(2*np.pi*fc*t) + np.random.randn(int(nSamps))
x = np.reshape(x,(nSampsPerFrame,nFrames),'F')

Xf = np.fft.fftshift(np.fft.fft(x[:,0]))

spect = np.zeros((nFFT,nFrames))
tFrame = nSampsPerFrame/fs
t_spect = np.arange(nFrames)*tFrame
fAxis = np.arange(int(-nFFT/2),int(nFFT/2))*fs/nFFT

for iFrame in np.arange(nFrames):
    spect[:,iFrame] = np.fft.fftshift(np.fft.fft(x[:,iFrame]))

maxValdB = np.max(mag2db(spect))


plt.figure()
plt.plot(t[:nSampsPerFrame],x[:,0])
plt.xlabel("time [sec]")
plt.ylabel("Amplitude")

plt.figure()
plt.plot(20*np.log10(np.abs(Xf)))

plt.figure()
# plt.imshow(mag2db(spect),aspect='auto',extent=(t[0]*1e3,t[-1]*1e3,fAxis[0]/1e3,fAxis[-1]/1e3))
plt.imshow(mag2db(spect),aspect='auto',vmin = maxValdB-30,vmax = maxValdB,
           extent= extentVals(1e3*t[[0,-1]],1e3*1/fs, 1e-3*fAxis[[0,-1]],1e-3*fs/nFFT))
