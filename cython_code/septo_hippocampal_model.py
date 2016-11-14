# -*- coding: utf-8 -*-
"""
septo-hippocampal model
"""

import lib2 as lib
import numpy as np
# import processingLib as plib
import matplotlib.pyplot as plt
#import os
import time

class SimmulationParams:
    def __init__(self, params=None, mode="default"):
        self.p = params
        self.mode = mode
      
    # variate frequency from septum 
    def iext_function(self, neuron_ind, compartment_name, t):
        return 0
          
        t = 0.001 * t
        Iext = 0
        if (neuron_ind >= 100 and neuron_ind < 105):
            Iext = np.cos(2 * np.pi * t * 8) + 1

        if (neuron_ind >= 105):
            Iext = np.cos(2 * np.pi * t * 8 + 2.65) + 1
        return Iext
               



cluster_pacemaker = {
    "V0": -50.0,
    "Iextmean": 1.2,        
    "Iextvarience": 0.5,
    "ENa": 50.0,
    "EK": -90.0,
    "El": -50.0,
    "Eh": -40.0,
    "gbarNa": 55.0,
    "gbarK": 8.0,
    "gbarKS": 12.0,
    "gbarH": 1.0,
    "gl": 0.1,
    "fi": 5,
}

fs_neuron = {
    "V0": -65.0,
    "Iextmean": 0.5,        
    "Iextvarience": 0.5,
    "ENa": 50.0,
    "EK": -90.0,
    "El": -65.0,
    "gbarNa": 55.0,
    "gbarK": 8.0,
    "gl": 0.1,   
    "fi": 10,
}

ext_synapse_params = {
    "Erev" : 60.0,
    "gbarS": 0.005,
    "alpha_s" : 1.1,
    "beta_s": 0.19,
    "K": 5.0,
    "teta": 2.0,
    "w" : 1.0,
    "delay" : 0,
}

inh_synapse_params = {
    "Erev" : -15.0,
    "gbarS": 0.005,
    "alpha_s" : 14.0,
    "beta_s": 0.07,
    "K": 2.0,
    "teta": 0.0,
    "w" : 1.0,
    "delay" : 0,
}

########## set model of septum ################
# number of neurons
Nglu = 40
NgabaCR = 40
NgabaPV1 = 40 
NgabaPV2 = 40
NgabaPacPV1 = 10
NgabaPacPV2 = 10

# number of synapses
Sparsness = 0.5


neurons = []
for idx in range(Nglu):
    neuron = {
        "type" : "FS_Neuron",
        "compartments" : fs_neuron.copy()
    }
    neuron["compartments"]["V0"] += 10 * np.random.rand()
    neuron["compartments"]["Iextmean"] = 0.5 # 0.5*np.random.randn()
    neurons.append(neuron)

for idx in range(NgabaCR):
    neuron = {
        "type" : "FS_Neuron",
        "compartments" : fs_neuron.copy()
    }
    neuron["compartments"]["V0"] += 10 * np.random.rand()
    neuron["compartments"]["Iextmean"] = 0 # 0.1*np.random.randn()
    neurons.append(neuron)

for idx in range(NgabaPV1):
    neuron = {
        "type" : "FS_Neuron",
        "compartments" : fs_neuron.copy()
    }
    neuron["compartments"]["V0"] += 10 * np.random.rand()
    neuron["compartments"]["Iextmean"] = 0.1 #0.5*np.random.randn()
    neurons.append(neuron)

for idx in range(NgabaPacPV1):
    neuron = {
        "type" : "ClusterNeuron",
        "compartments" : cluster_pacemaker.copy()
    }
    neuron["compartments"]["V0"] += 10 * np.random.rand()
    neuron["compartments"]["Iextmean"] = 0.1 # 0.5*np.random.randn()
    neurons.append(neuron)

for idx in range(NgabaPV2):
    neuron = {
        "type" : "FS_Neuron",
        "compartments" : fs_neuron.copy()
    }
    neuron["compartments"]["V0"] += 10 * np.random.rand()
    neuron["compartments"]["Iextmean"] = 0.7 #0.5*np.random.randn()
    neurons.append(neuron)

for idx in range(NgabaPacPV2):
    neuron = {
        "type" : "ClusterNeuron",
        "compartments" : cluster_pacemaker.copy()
    }
    neuron["compartments"]["V0"] += 10 * np.random.rand()
    neuron["compartments"]["Iextmean"] = 0.7 #0.5*np.random.randn()
    neurons.append(neuron)


synapses = []
# synapse from Glu to Glu neurons

for idx1 in range(Nglu):
    for idx2 in range(Nglu):
        if (Sparsness < np.random.rand() or idx1 == idx2):
            continue
        pre_ind = idx1
        post_ind = idx2
        synapse = {
            "type" : "ComplexSynapse",
            "pre_ind": pre_ind, 
            "post_ind": post_ind,
            "pre_compartment_name": "soma",
            "post_compartment_name" : "soma",
            "params": ext_synapse_params.copy(),
        }
        synapse["params"]["w"] = 1
        synapses.append(synapse)

# synapses from Glu naurons to GABA(CR) neurons
for idx1 in range(Nglu):
    for idx2 in range(Nglu, Nglu+NgabaCR):
        if (Sparsness < np.random.rand() or idx1 == idx2):
            continue
        pre_ind = idx1
        post_ind = idx2
        synapse = {
            "type" : "ComplexSynapse",
            "pre_ind": pre_ind, 
            "post_ind": post_ind,
            "pre_compartment_name": "soma",
            "post_compartment_name" : "soma",
            "params": ext_synapse_params.copy(),
        }
        synapse["params"]["w"] = 0.2
        synapses.append(synapse)

# synapses from GABA(CR) neurons to Glu naurons
for idx1 in range(Nglu, Nglu+NgabaCR):
    for idx2 in range(Nglu):
        if (idx1 == idx2):
            continue

        pre_ind = idx1
        post_ind = idx2
        synapse = {
            "type" : "ComplexSynapse",
            "pre_ind": pre_ind, 
            "post_ind": post_ind,
            "pre_compartment_name": "soma",
            "post_compartment_name" : "soma",
            "params": inh_synapse_params.copy(),
        }
        synapse["params"]["w"] = 1.5
        synapses.append(synapse)
        
       
# synapses from Glu neurons to PV1 and pac-PV1 naurons  
for idx1 in range(0, Nglu):
    for idx2 in range(Nglu+NgabaCR, Nglu+NgabaCR+NgabaPV1+NgabaPacPV1):
        if (Sparsness < np.random.rand() or idx1 == idx2):
            continue
        pre_ind = idx1
        post_ind = idx2
        synapse = {
            "type" : "ComplexSynapse",
            "pre_ind": pre_ind, 
            "post_ind": post_ind,
            "pre_compartment_name": "soma",
            "post_compartment_name" : "soma",
            "params": ext_synapse_params.copy(),
        }
        synapse["params"]["w"] = 1
        synapses.append(synapse)

# synapses from PV1 and pac-PV1 naurons to PV2 and pac-PV2
for idx1 in range(Nglu+NgabaCR, Nglu+NgabaCR+NgabaPV1+NgabaPacPV1):
    for idx2 in range(Nglu+NgabaCR+NgabaPV1+NgabaPacPV1, Nglu+NgabaCR+NgabaPV1+NgabaPacPV1+NgabaPV2+NgabaPacPV2):
        if (Sparsness < np.random.rand() or idx1 == idx2):
            continue
        pre_ind = idx1
        post_ind = idx2
        synapse = {
            "type" : "ComplexSynapse",
            "pre_ind": pre_ind, 
            "post_ind": post_ind,
            "pre_compartment_name": "soma",
            "post_compartment_name" : "soma",
            "params": inh_synapse_params.copy(),
        }
        synapse["params"]["w"] = 0.2
        synapses.append(synapse) 
    
# synapses from PV2 and pac-PV2 naurons to PV1 and pac-PV1  
for pre_ind in range(Nglu+NgabaCR+NgabaPV1+NgabaPacPV1, Nglu+NgabaCR+NgabaPV1+NgabaPacPV1+NgabaPV2+NgabaPacPV2):
    for post_ind in range(Nglu+NgabaCR, Nglu+NgabaCR+NgabaPV1+NgabaPacPV1):
        if (Sparsness < np.random.rand() or idx1 == idx2):
            continue
        synapse = {
            "type" : "ComplexSynapse",
            "pre_ind": pre_ind, 
            "post_ind": post_ind,
            "pre_compartment_name": "soma",
            "post_compartment_name" : "soma",
            "params": inh_synapse_params.copy(),
        }
        synapse["params"]["w"] = 0.1
        synapses.append(synapse) 


for s in synapses:
    s["params"]["w"] *= 2

################################
sim = SimmulationParams()
net = lib.Network(neurons, synapses)
curent_time = time.time()
net.integrate(0.1, 500, sim.iext_function)
print ("Calculation time is %0.3f sec" % (time.time() - curent_time) )
firing = net.getFiring()


plt.plot(firing[0, :], firing[1, :], ".")
plt.ylim(0, 181)
plt.show()
