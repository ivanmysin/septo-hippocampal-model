# -*- coding: utf-8 -*-
"""
main script for modeling
"""
import lib
import numpy as np
import matplotlib.pyplot as plt 
import time


dt = 0.1
duration = 1000

pac_in_model = True
nglu = 40
n_glu_pac = 0

ngaba_cr = 40
n_gaba_cr_pac = 0

if (pac_in_model):
	ngaba_pv1 = 40
	ngaba_pv2 = 40
	n_gaba_pv1_pac = 10
	n_gaba_pv2_pac = 10 
else:
	ngaba_pv1 = 50
	ngaba_pv2 = 50
	n_gaba_pv1_pac = 0
	n_gaba_pv2_pac = 0 


ngaba_pv1 = ngaba_pv1 + n_gaba_pv1_pac
ngaba_pv2 = ngaba_pv2 + n_gaba_pv2_pac
ngaba_cr = ngaba_cr + n_gaba_cr_pac
nglu = nglu + n_glu_pac

ngaba = ngaba_cr + ngaba_pv1 + ngaba_pv2

nn = nglu + ngaba # nn is the whole number of neurons

gaba_pv1 = (np.arange(ngaba_pv1)).astype(int)
gaba_pv2 = (np.arange(ngaba_pv2) + ngaba_pv1).astype(int)
gaba_cr = (np.arange(ngaba_cr) + ngaba_pv1 + ngaba_pv2).astype(int)
glu = (np.arange(nglu) + ngaba_cr + ngaba_pv1 + ngaba_pv2).astype(int)
gaba = (np.append(gaba_pv1, gaba_pv2)).astype(int)
gaba = (np.append(gaba, gaba_cr)).astype(int)

if (n_gaba_pv1_pac > 0):
	gaba_pv1_pac = gaba_pv1[0:n_gaba_pv1_pac]
else:
	gaba_pv1_pac = np.array([])

if (n_gaba_pv2_pac > 0): 
	gaba_pv2_pac = gaba_pv2[0:n_gaba_pv2_pac]
else:
	gaba_pv2_pac = np.array([])


if (n_glu_pac > 0):
	glu_pac = glu[0:n_glu_pac]
else: 
	glu_pac = np.array([])



pac = np.append(gaba_pv1_pac, gaba_pv2_pac)
pac = np.append(pac, glu_pac)
pac = pac.astype(int)

# parametrs of neurons
ENa = 55 # mv Na reversal potential
EK = -85 # mv K reversal potential
El = np.zeros(nn) - 65 # mv Leakage reversal potential
El[pac] = -50
Eh = -40
gbarNa = 50    # mS/cm^2 Na conductance
gbarK=8        # mS/cm^2 K conductance
gbarKS = np.zeros(nn)
gbarKS[pac] = 12
        
gbarH = np.zeros(nn)
gbarH[pac] = 1     # 0.15;
gl = 0.1 
fi = np.zeros(nn) + 10
fi[pac] = 5

V0 = np.random.rand(nn)*20 - 55
varience_iext = 0.1
Iextmean = np.zeros(nn) #$varience_iext * grandom($nn);
Iextmean[glu] += 0.3
Iextmean[gaba_pv2] += 0.7
Iextmean[gaba_pv1] += 0.3 # $Iextmean($gaba_pv2)*$Textratio;


disp = np.zeros(nn) + 0.1
disp[gaba_cr] = 0.1


########################################
# parameters of synapses
variance_w = 1
w = np.zeros([nn, nn]) #  W is the matrix of synaptic weights
# $w($gaba_cl, $gaba_cl) .= 1;
w[ gaba_cr[0]:gaba_cr[-1], glu[0]:glu[-1] ] = 1.5
w[glu[0]:glu[-1], gaba_cr[0]:gaba_cr[-1]] = 0.2
w[glu[0]:glu[-1], glu[0]:glu[-1]] = 1
        
w[glu[0]:glu[-1], gaba_pv1[0]:gaba_pv1[-1]] = 1
w[gaba_cr[0]:gaba_cr[-1], gaba_pv1[0]:gaba_pv1[-1]] = 1
        
w[gaba_pv1[0]:gaba_pv1[-1], gaba_pv2[0]:gaba_pv2[-1]] = 0.2
w[gaba_pv2[0]:gaba_pv2[-1], gaba_pv1[0]:gaba_pv1[-1]] = 0.1
np.fill_diagonal(w, 0)
w *= 20

teta = np.zeros(nn)
teta[glu] = 2
K = np.zeros(nn) + 2
Erev = np.zeros(nn) - 75
Erev[glu] = 0
        
alpha_s = np.zeros(nn) + 14
alpha_s[glu] = 1.1
beta_s = np.zeros(nn) + 0.07
beta_s[glu] = 0.19
        
gsyn = np.zeros(nn) + 0.0005
gsyn[glu] = 0.0005



if __name__ == "__main__":
    
    neurons_properties = []

    for i in range(nn):
    	#print model "$V[$i], $ENa, $EK,$El[$i],$gbarNa, $gbarK, $gl,  $fi[$i], $Iextmean[$i],  $disp[$i]";
    	neuron_params = [V0[i], ENa, EK, El[i], gbarNa, gbarK, gl, fi[i], Iextmean[i], disp[i]]
    	if (gbarKS[i] != 0 and gbarH[i] != 0):
         neuron_params.append(Eh)
         neuron_params.append(gbarKS[i])
         neuron_params.append(gbarH[i])
    	neurons_properties.append(neuron_params)

    synapses_propeties = []
    for i in range(nn):
        w_ = w[i, :]
        for j in range(nn):
            p = np.random.rand()
            if (w[i, j] > 0 and p >= 0.5):
                synapse_params = [i, j, Erev[i], gsyn[i], w[i, j], alpha_s[i], beta_s[i], teta[i], K[i]]
                synapses_propeties.append(synapse_params)

    t = time.time()
    septum = lib.Network(neurons_properties, synapses_propeties)
    septum.integrate(dt, duration)
    print (time.time() - t)
    firing = septum.getRaster()

    plt.plot(firing[:, 0], firing[:, 1], ".")

