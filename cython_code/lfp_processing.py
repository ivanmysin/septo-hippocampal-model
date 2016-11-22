# -*- coding: utf-8 -*-
"""
main script for processing
"""

import numpy as np
import matplotlib.pyplot as plt
import processingLib as plib
import os
def make_figures(path, file, septumInModel = True):
    Np = 400    # number of pyramide neurons
    Nb = 50     # number of basket cells
    Nolm = 50   # number of olm cells
    
    if (septumInModel):
        Nglu = 40
        NgabaCR = 40
        NgabaPV1 = 40 
        NgabaPV2 = 40
        NgabaPacPV1 = 10
        NgabaPacPV2 = 10
    else:
        NSG = 0    # number of spike generators of


   
    saving_path = path + file[0]
    
    file = path + file
    
    fd = 10000
    
    res = np.load(file)
    
    lfp = res[()]["results"]["lfp"]
    #V = res[()]["results"]["V"]
    firing = res[()]["results"]["firing"]
    
    lfp_filtred = plib.filtrate_lfp(lfp, fd)
    lfp_filtred = lfp_filtred[0:-1:20] 
    fd = 500
    t = np.linspace(0, lfp_filtred.size/fd, lfp_filtred.size)
    
    # calculate and plot wavelet spectrum and LFP signal
    freqs, coefs = plib.computemycwt(fd, lfp_filtred)
    
    plt.figure()
    plt.subplot(211)
    Z = np.abs(coefs)
    plt.pcolor(t, freqs, Z, cmap='rainbow', vmin=0, vmax=Z.max())
    plt.title('Wavelet spectrum of simulated LFP')
    # set the limits of the plot to the limits of the data
    plt.axis([t[0], t[-1], freqs[0], freqs[-1]])
    plt.colorbar()
    
    
    plt.subplot(212)
    plt.plot(t, lfp_filtred, color="black", linewidth=2)
    plt.xlim(t[0], t[-1])
    #plt.ylim(1.2*lfp_filtred.max(), -1.2*lfp_filtred.min())
    plt.colorbar()
    plt.savefig(saving_path + "wavelet.png", dpi = 500)
    
    # calculate and plot asymmetry index
    asymmetry_index, idx_max, idx_min = plib.get_asymetry_index(lfp_filtred)
    if (asymmetry_index.size > 0):
        plt.figure()
        plt.subplot(121)
        plt.plot(t, lfp_filtred, color="black", linewidth=2)
        lfp_max = lfp_filtred.max()
        lfp_min = lfp_filtred.min()
        for i in idx_max:
            plt.plot([t[i], t[i]], [lfp_min, lfp_max], color="blue", linestyle="dashed", linewidth=2)
    
        for i in idx_min:
            plt.plot([t[i], t[i]], [lfp_min, lfp_max], color="blue", linestyle="dashed", linewidth=2)
            
        plt.subplot(122)    
        plt.hist(asymmetry_index, 10, normed=1, facecolor='blue', alpha=0.75)
        plt.xlim(-1.5, 1.5)
        plt.xlabel('Asymmetry index')
        plt.ylabel('Theta cycles')
        plt.title('')
        plt.savefig(saving_path + "asymmetry_index.png", dpi = 500)
    else:
        print ("Zero elements in asymemetry index array")
    
    # calculate and plot amplitude - phase cross frequency coupling between theta and gamma rhythms
    ampFrs = np.linspace(15, 80, 65)
    phFrs = np.array([4.0, 12.0])
    apcoupling = plib.cossfrequency_phase_amp_coupling(lfp_filtred, fd, ampFrs, phFrs, 0.1)
    theta_phase = np.linspace(-np.pi, np.pi, apcoupling.shape[1])
    plt.figure()
    plt.subplot(111)
    plt.pcolor(theta_phase, ampFrs, apcoupling, cmap='rainbow', vmin=apcoupling.min(), vmax=apcoupling.max())
    plt.title('Cross frequency coupling between theta and gamma rhythms')
    plt.axis([-np.pi, np.pi, ampFrs[0], ampFrs[-1]])
    plt.colorbar()
    plt.savefig(saving_path + "amp_phase_coupling.png", dpi = 500)
    
    # calculate and plot phase-phase cross frequency coupling between theta and gamma rhythms
    phFrs1 = np.array([4, 12])
    phFrs2 = np.array([30, 90])
    nmarray = np.ones([2, 12])
    nmarray[1, :] = np.arange(1, 13)
    ppcoupling = plib.cossfrequency_phase_phase_coupling (lfp_filtred, fd, phFrs1, phFrs2, nmarray)
    
    plt.figure()
    plt.plot(nmarray[1, :], ppcoupling)
    plt.xlim(1, nmarray[1, -1])
    plt.savefig(saving_path + "phase_phase_coupling.png", dpi = 500)
    
    # plot raster of neuronal activity
    firing_slices = {}
    plt.figure()
    firing[0, :] *= 0.001
    # plot septal neurons
    if (septumInModel):
        cum_it = Nglu
        sl = firing[1, :] <= cum_it
        firing_slices["glu"] = np.copy(sl)
        glu_line = plt.scatter(firing[0, sl], firing[1, sl], color="r")
        
        sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + NgabaCR)
        firing_slices["gaba_cr"] = np.copy(sl)
        cr_line = plt.scatter(firing[0, sl], firing[1, sl], color="g")
        cum_it += NgabaCR
        
        sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + NgabaPV1 + NgabaPacPV1)
        firing_slices["gaba_pv1"] = np.copy(sl)
        pv1_line = plt.scatter(firing[0, sl], firing[1, sl], color="b")
        cum_it += NgabaPV1 + NgabaPacPV1
        
        sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + NgabaPV2 + NgabaPacPV2)
        firing_slices["gaba_pv2"] = np.copy(sl)
        pv2_line = plt.scatter(firing[0, sl], firing[1, sl], color="b")
        cum_it += NgabaPV2 + NgabaPacPV2

        # plot hippocampal neurons
        sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + Np)
    else:
        cum_it = Np    
        sl = (firing[1, :] < cum_it)
    firing_slices["pyramide"] = np.copy(sl)
    pyr_line = plt.scatter(firing[0, sl], firing[1, sl], color="c")
    cum_it += Np
    
        
    sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + Nb)
    firing_slices["basket"] = np.copy(sl)
    basket_line = plt.scatter(firing[0, sl], firing[1, sl], color="k")
    cum_it += Nb
    sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + Nolm)
    firing_slices["olm"] = np.copy(sl)
    olm_line = plt.scatter(firing[0, sl], firing[1, sl], color="m")
    
    if not(septumInModel):
        cum_it += Nolm
        sl = (firing[1, :] > cum_it)
                
        pv1_line = plt.scatter(firing[0, sl], firing[1, sl], color="b")
        pv2_line = plt.scatter(firing[0, sl], firing[1, sl], color="b")
    
    if (septumInModel):          
        plt.legend((glu_line, cr_line, pv1_line, pv2_line, pyr_line, basket_line, olm_line),
               ('Glu', 'GABA(CR)', 'GABA(PV1)', 'GABA(PV2)', 'Pyramide', 'Basket', 'OLM'),
               scatterpoints=1,
               loc='upper left',
               ncol=1,
               fontsize=12)
    else:
        plt.legend((pv1_line, pv2_line, pyr_line, basket_line, olm_line),
               ('GABA(PV1)', 'GABA(PV2)', 'Pyramide', 'Basket', 'OLM'),
               scatterpoints=1,
               loc='upper left',
               ncol=1,
               fontsize=12)
          
          
               
               
    plt.xlim(0, t[-1])
    plt.ylim(0, cum_it + Nolm + 100)
    plt.tight_layout()
    plt.savefig(saving_path + "raster.png", dpi = 500)
    
    # calculate and plot phase coupling between units activity and theta rhythm 
    plt.figure()
    plt.subplot(111, polar=True)
    theta_lfp = plib.butter_bandpass_filter(lfp_filtred, 4, 12, fd, 2)
    
    color = "y"
    for key, sl in firing_slices.items():
        fir = firing[:, sl]
        if (fir.shape[0] == 0):
            continue
        

        angles, length = plib.get_units_phase_coupling(theta_lfp, fir, fd)
        #angles += np.pi
        
        if (key == "glu"):
            color = "r"
        if (key == "gaba_cr"):
            color = "g"
        if (key == "gaba_pv1" or key == "gaba_pv2"):
            color = "b"
        if (key == "pyramide"):
            color = "c"
        if (key == "basket"):
            color = "k"  
        if (key == "olm"):
            color = "m" 
        plt.scatter(angles, length, color=color)
        
        # calculate histogram ?
    """
    plt.legend((glu_line, cr_line, pv1_line, pv2_line, pyr_line, basket_line, olm_line),
               ('Glu', 'GABA(CR)', 'GABA(PV1)', 'GABA(PV2)', 'Pyramide', 'Basket', 'OLM'),
               scatterpoints=1,
               loc='upper left',
               ncol=1,
               fontsize=12)
    """
    
    plt.savefig(saving_path + "phase_disribution_of_neurons.png", dpi = 500)
    plt.figure()
    plt.subplot(111)
    theta_lfp = plib.butter_bandpass_filter(lfp_filtred, 4, 12, fd, 2)
    
    color = "y"
    for key, sl in firing_slices.items():
        fir = firing[:, sl]
        if (fir.size == 0):
            continue
        angles, length = plib.get_units_phase_coupling(theta_lfp, fir, fd)
        
        minn = np.min( firing[1, sl] ) 
        maxn = np.max( firing[1, sl] ) 
        numbers = np.linspace(minn, maxn, angles.size)
        
        if (key == "glu"):
            color = "r"
        if (key == "gaba_cr"):
            color = "g"
        if (key == "gaba_pv1" or key == "gaba_pv2"):
            color = "b"
        if (key == "pyramide"):
            color = "c"
        if (key == "basket"):
            color = "k"  
        if (key == "olm"):
            color = "m" 
        plt.scatter(angles, numbers, color=color)
        
        # calculate histogram ?
    plt.xlim(-np.pi, np.pi)
    plt.savefig(saving_path + "phase_disribution_of_neurons2.png", dpi = 500)
    """
    plt.legend((glu_line, cr_line, pv1_line, pv2_line, pyr_line, basket_line, olm_line),
               ('Glu', 'GABA(CR)', 'GABA(PV1)', 'GABA(PV2)', 'Pyramide', 'Basket', 'OLM'),
               scatterpoints=1,
               loc='upper left',
               ncol=1,
               fontsize=12)
    """
####################################
"""
path = "/home/ivan/Data/modeling_septo_hippocampal_model/septo_hippocampal_model2/without_hipp-septum_input/"
path = "/home/ivan/Data/modeling_septo_hippocampal_model/septo_hippocampal_model2/OLM2PV1/"
path = "/home/ivan/Data/modeling_septo_hippocampal_model/septo_hippocampal_model2/OLM2PV1andPV2/"
path = "/home/ivan/Data/modeling_septo_hippocampal_model/septo_hippocampal_model2/OLM2PV2/"
"""
path = "/home/ivan/Data/modeling_septo_hippocampal_model/hippocampal_model/only_one_rhytm/"
for file in os.listdir(path):
    if (file[0] == "." or file[-4:] != ".npy"):
        continue
    make_figures(path, file, False)

"""
# for seizures analisis
path = "/home/ivan/Data/modeling_septo_hippocampal_model/seisures/norm/"
file = "_all_results.npy"
make_figures(path, file)
"""