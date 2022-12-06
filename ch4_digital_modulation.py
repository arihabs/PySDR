# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 10:01:31 2022

@author: ArielHabshush
"""

import numpy as np
import matplotlib.pyplot as plt

def bin2gray(binNum):
    return binNum ^ (binNum >> 1)

def qpskMod(syms,coding="gray"):
    if(coding == "gray"):
        syms = bin2gray(syms)

    return np.exp(1j*np.pi/4*(1+2*syms))

def rms(x):
    return np.sqrt(np.sum(abs(x)**2)/len(x))

def powDb(x):
    return 20*np.log10(rms(x))


# QPSK
nSyms = int(1e3)
bitsPerSym = 2
nBits = nSyms * bitsPerSym

# bitStream = np.random.randint(0,2,nBits).reshape(bitsPerSym,nSyms)
# Random bit stream
bitStream = np.random.randint(0,2,nBits).astype(str).reshape(nSyms,bitsPerSym)

# Convert bits to symbols
# Concatenate each row of strings into a single string and convert from binary to decimal
syms =  np.array([int(''.join(list(bitStream[row])),2) for row in range(bitStream.shape[0])],dtype=('int'))
symsIQ = qpskMod(syms,"bin")

# Add noise
snr = 20
noisePowDb = -snr
noiseVar = 10**(noisePowDb/10)
noiseSig = np.sqrt(noiseVar/2)*(np.random.randn(nSyms) + 1j*np.random.randn(nSyms))

symsIQ_noisy = symsIQ + noiseSig
# print(f"Noise Power = {20*np.log10(rms(noiseSig)):{1}.{5}} dB")
print(f"Noise Power = {powDb(noiseSig)}, SigPow = {powDb(symsIQ)}, SNR = {powDb(symsIQ)-powDb(noiseSig)}")
#%% Figures
plt.figure()
plt.plot(symsIQ.real,symsIQ.imag,'.')
plt.grid()

plt.xlabel("I",fontsize=14)
plt.ylabel("Q",fontsize=14)

plt.plot(symsIQ_noisy.real,symsIQ_noisy.imag,'.')


# bitStream2 = np.zeros(int(nSyms), dtype="S2")
# # for col in range(bitStream.shape[1]):
# for row in range(bitStream.shape[0]):
#     bitStream2[row] = np.array( int(''.join(list(bitStream[row])),2),dtype='int')
#     # print(bitStream2[row])

# print(bitStream2)


