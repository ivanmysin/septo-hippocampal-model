# -*- coding: utf-8 -*-
"""
test
"""

import ca1
import numpy as np
import matplotlib.pyplot as plt
soma_params = {
    "V0": 0.0,
    "C" : 3.0,
    "Iextmean": 4.5,        
    "Iextvarience": 0.9,
    "ENa": 120.0,
    "EK": -15.0,
    "El": 0.0,
    "ECa": 140.0,
    "CCa": 0.05,
    "sfica": 0.13,
    "sbetaca": 0.075, 
    "gbarNa": 30.0,
    "gbarK_DR": 17.0,
    "gbarK_AHP": 0.8,        
    "gbarK_C " : 15.0,
    "gl": 0.1,
    "gbarCa": 6.0,
}

dendrit_params = {
    "V0": 0.0,
    "C" : 3.0,
    "Iextmean": 4.5,        
    "Iextvarience": 0.9,
    "ENa": 120.0,
    "EK": -15.0,
    "El": 0.0,
    "ECa": 140.0,
    "CCa": 0.05,
    "sfica": 0.13, 
    "sbetaca": 0.075, ######## ??????????
    "gbarNa": 0.0,
    "gbarK_DR": 0.0,
    "gbarK_AHP": 0.8,        
    "gbarK_C " : 5.0,
    "gl": 0.1,
    "gbarCa": 5.0,
}


olm_params = {
    "V0": -60,
    "C" : 1.0,
    "Iextmean": -0.5, #### -9.5 !!!!!!!!
    "Iextvarience": 0.8,
    "ENa": 55.0,
    "EK": -90.0,
    "El": -65.0,
    "EH": -20, 
    "gbarNa": 52.0,
    "gbarNap": 0.5,
    "gbarK_DR": 11.0,
    "gbarH": 1.45,
    "gl": 0.5,
}


ext_synapce_params = {
    "Erev" : 60.0,
    "gbarS": 0.005,
    "tau" : 2.0,
    "w" : 1.0,
}

inh_synapce_params = {
    "Erev" : -10.0,
    "gbarS": 0.01,
    "tau" : 2.0,
    "w" : 50.0,
}

basket_params = {
     "V0": -60.0,
     "Iextmean": 0.5,        
     "Iextvarience": 0.9,
     "ENa" : 50.0, 
     "EK": -90.0,
     "El": -65.0,
     "gbarNa": 55.0,
     "gbarK": 8,
     "gl": 0.1,   
     "fi" : 10,
}
################################
npyr = 20
pyramide_neurons = []
nsynapses_pyr_pyr = 30
synapses_pyr_pyr = []
nolm = 10
olms = []

nsynapses_olm_pyr = 50
synapses_olm_pyr = []

nbaskets = 10
baskets = []
nsynapses_basket_pyr = nsynapses_olm_pyr
synapses_basket_pyr = []


for idx in range(npyr):
    soma = ca1.Compartment(soma_params)
    dendrite = ca1.Compartment(dendrit_params) 
    connection = ca1.IntercompartmentConnection(soma, dendrite, 1.5, 0.5)
    neuron = ca1.ComplexNeuron([soma, dendrite], [connection])
    pyramide_neurons.append(neuron)

for idx in range(nsynapses_pyr_pyr):
    pre_ind = np.random.randint(0, npyr)
    post_ind =  np.random.randint(0, npyr)
    synapse = ca1.Synapse(pyramide_neurons[pre_ind], pyramide_neurons[post_ind].getSoma() , ext_synapce_params)
    synapses_pyr_pyr.append(synapse)


for idx in range(nolm):
    olm = ca1.OLMcell(olm_params)
    olms.append(olm)

for idx in range(nsynapses_olm_pyr):
    pre_ind = np.random.randint(0, nolm)
    post_ind =  np.random.randint(0, npyr)
    synapse = ca1.Synapse(olms[pre_ind], pyramide_neurons[post_ind].getDendrite() , inh_synapce_params)
    synapses_olm_pyr.append(synapse)

for idx in range(nbaskets):
    basket = ca1.FS_neuron(basket_params)
    baskets.append(basket)

for idx in range(nsynapses_basket_pyr):
    pre_ind = np.random.randint(0, nbaskets)
    post_ind =  np.random.randint(0, npyr)
    synapse = ca1.Synapse(baskets[pre_ind], pyramide_neurons[post_ind].getSoma() , inh_synapce_params)
    synapses_basket_pyr.append(synapse)




dt = 0.01
dur = 1000
t = 0


while (t < dur):
    for idx in range(npyr):
        pyramide_neurons[idx].integrate(dt, dt)
    
    Isept = np.cos(2 * np.pi * t * 4) - 5.5
    for idx in range(nolm):
        
        olms[idx].setIsyn(Isept)
        
        olms[idx].integrate(dt, dt)
    
    Isept = np.cos(2 * np.pi * t * 4 + np.deg2rad(152)) - 5.5
    for idx in range(nbaskets):
       
        baskets[idx].setIsyn(Isept)
         
        baskets[idx].integrate(dt, dt)
        
        
    for idx in range(nsynapses_pyr_pyr):
        synapses_pyr_pyr[idx].integrate(dt)
        
    for idx in range(nsynapses_olm_pyr):
        synapses_olm_pyr[idx].integrate(dt)
        
    for idx in range(nsynapses_basket_pyr):
        synapses_basket_pyr[idx].integrate(dt)
    

    t += dt


lfp = 0
for idx in range(npyr):
    lfp += np.array(pyramide_neurons[idx].getSoma().getVhist())
lfp /= idx
Fd = 1000 // dt
plt.figure()
Vs = np.array(pyramide_neurons[0].getSoma().getVhist())
Vd = np.array(pyramide_neurons[0].getDendrite().getVhist())
t = np.linspace(0, dur, Vs.size)
plt.plot(t, Vs, "b", t, Vd, "g")

lfp_ = lfp[t >= 500]
plt.figure()
plt.plot(t, lfp)


plt.figure()
Pxx, freqs, bins, im = plt.specgram(lfp, NFFT=1024, Fs=Fd, noverlap=900,
                                cmap=plt.cm.gist_heat)
plt.ylim(0, 60)

plt.figure()
plt.magnitude_spectrum(lfp_, Fs=Fd)
plt.xlim(0, 60)