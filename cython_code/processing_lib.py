# -*- coding: utf-8 -*-
"""
library for processing results
"""
import numpy as np
import processingLib as plib

def filtrate_lfp(lfp, fd):
    lfp_mean = np.mean(lfp)
    lfp_std = np.std(lfp)
    lfp_norm = (lfp - lfp_mean) / lfp_std
    
    lfp_fft = 2 * np.fft.rfft(lfp_norm) / lfp.size
    
    w = np.fft.rfftfreq(lfp.size, 1/fd )
    Z = np.exp(-0.05*w)
    Z = 0.8*Z + 0.2
    Z[w>=80] = 0
    
    lfp_fft *= Z
    
    lfp_filtred = (np.fft.irfft(lfp_fft) + lfp_mean)*lfp_std
    
    return lfp_filtred