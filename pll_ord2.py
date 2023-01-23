# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 16:40:26 2023

@author: ArielHabshush
"""
import math
import cmath

def phase_detector(in1, in2, detectType = "arg"):
    if detectType == "arg":
        mix = in1 * in2.conjugate()
        out = math.atan2(mix.imag, mix.real)
    elif detectType == "im":
        pass
    else:
        pass
    return out

def loop_filter_pi(in1,lf_sum2, k1 = 0.1479, k2 = 0.0059):
    lf_sum2_d = lf_sum2
    lf_gain_1 = k1*in1
    lf_gain_2 = k2*in1
    lf_sum2 = lf_sum2_d + lf_gain_2
    out = (lf_gain_1 + lf_sum2, lf_sum2)
    return out

def dds(in1, freq, dds_sum1):
    dds_sum1_d = dds_sum1
    dds_sum1 = in1 + freq + dds_sum1_d
    dds_out = cmath.exp(1j*dds_sum1_d)
    out = (dds_out,dds_sum1)
    return out

def pll_ord2(in1,freq, states=None):
    i = 0
    pd_in2 = cmath.exp(1j*0)
    lf_sum2 = 0
    dds_sum1 = 0
    Nsamps = len(in1)
    pll_out = []
    err = []
    lf_out_all = []
    while(i < Nsamps):
        phaseOut = phase_detector(in1[i], pd_in2)
        lf_out, lf_sum2 = loop_filter_pi(phaseOut,lf_sum2)
        dds_out, dds_sum1 = dds(lf_out, freq, dds_sum1)
        pd_in2 = dds_out
        pll_out.append(dds_out)
        err.append(phaseOut)
        lf_out_all.append(lf_out)
        i+=1
    return (pll_out,err)
if(__name__ == '__main__'):
    import numpy as np
    import matplotlib.pyplot as plt

    Nsamps = 1000
    freqNorm = 1/10
    radFreq = 2*cmath.pi*freqNorm
    phase = cmath.pi

    DEBUG_DDS = False

    if DEBUG_DDS:
        dds_sum1 = 0
        dds_out_all = []
        in1 = 0
        freq = radFreq
        for i in range(Nsamps):
            dds_out, dds_sum1 = dds(in1, freq, dds_sum1)
            dds_out_all.append(dds_out)

        plt.figure()
        plt.plot(np.array(dds_out_all).real)
        plt.plot(np.array(dds_out_all).imag)


    pll_in = np.array([cmath.exp(1j*(radFreq*n + phase)) for n in range(Nsamps)])

    pll_out, err = pll_ord2(pll_in,radFreq)


    plt.figure()
    plt.plot(pll_in.real)
    plt.plot(np.array(pll_out).real)
    plt.grid()
    plt.title("pll_in and pll_out")
    plt.legend(["pll in", "pll_out"])

    plt.figure()
    plt.plot(err,"-*")



    print("test")
