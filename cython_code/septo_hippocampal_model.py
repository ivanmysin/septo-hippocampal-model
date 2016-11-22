# -*- coding: utf-8 -*-
"""
septo-hippocampal model
"""

import lib2 as lib
import numpy as np
#import matplotlib.pyplot as plt
import os
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
               

def run_model(path, experiment):

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
    
    soma_params = {
        "V0": 0.0,
        "C" : 3.0,
        "Iextmean": -0.5,        
        "Iextvarience": 0.2,
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
        
    dendrite_params = {
        "V0": 0.0,
        "C" : 3.0,
        "Iextmean": -0.5,        
        "Iextvarience": 0.2,
        "ENa": 120.0,
        "EK": -15.0,
        "El": 0.0,
        "ECa": 140.0,
        "CCa": 0.05,
        "sfica": 0.13, 
        "sbetaca": 0.075, 
        "gbarNa": 0.0,
        "gbarK_DR": 0.0,
        "gbarK_AHP": 0.8,        
        "gbarK_C " : 5.0,
        "gl": 0.1,
        "gbarCa": 5.0,
    }
        
    connection_params = {
        "compartment1": "soma",
        "compartment2": "dendrite",
        "p": 0.5,
        "g": 1.5,
    }
        
    basket_fs_neuron = {
        "V0": -65.0,
        "Iextmean": 0.2,        
        "Iextvarience": 0.2,
        "ENa": 50.0,
        "EK": -90.0,
        "El": -65.0,
        "gbarNa": 55.0,
        "gbarK": 8.0,
        "gl": 0.1,   
        "fi": 10,
    }
        
    olm_params = {
        "V0" : -70,
        "gl" : 0.05,
        "El" : -70,
        "gbarNa" : 10.7, 
        "ENa" : 90,
        "gbarK" : 31.9,
        "EK" : -100,
        "gbarKa" : 16.5,
        "gbarH" : 0.05,
        "EH" : -32.9,
        "Iextmean" : 0.2,        
        "Iextvarience": 0.2,
    }
    
    
    ########## set model of septum ################
    # number of neurons
    Nglu = 40
    NgabaCR = 40
    NgabaPV1 = 40 
    NgabaPV2 = 40
    NgabaPacPV1 = 10
    NgabaPacPV2 = 10
    
    
    NNseptum = Nglu + NgabaCR + NgabaPV1 + NgabaPV2 + NgabaPacPV1 + NgabaPacPV2
    
    Sparsness = 0.5 # sparsness in septal network
    
    
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
            synapse["params"]["w"] = 0.5
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
        
    ###### set model of hippocampus ####################
    # number of neurons
    Np = 400 # number of pyramide neurons
    Nb = 50   # number of basket cells
    Nolm = 50 # number of olm cells
    
    # Number synapses
    Ns = 400 # number synapses between pyramide cells
    Nint2pyr = 50 # 4 # number synapses from one interneuron to one pyramide neuron 
    
    # NSG = 20 # number of spike generators
    Ns2in = 200 # number synapses from septal generators to hippocampal interneurons 
    NsOLM2PV = 400 # number synapses from OLM cell to MSDB PV1 cells 
        
    for idx in range(Np):
        soma = {"soma" : soma_params.copy()}
        soma["soma"]["V0"] = 0.5 * np.random.randn()
        soma["soma"]["Iextmean"] += 0.5*np.random.randn()
            
        dendrite = {"dendrite" : dendrite_params.copy()}
        dendrite["dendrite"]["V0"] = soma["soma"]["V0"]
        dendrite["dendrite"]["Iextmean"] += 0.5*np.random.randn()
           
        connection = connection_params.copy()
        neuron = {
            "type" : "pyramide",
            "compartments" : [soma, dendrite],
            "connections" : [connection]
        }
        neurons.append(neuron)
        
    for idx in range(Nb):
        neuron = {
            "type" : "FS_Neuron",
            "compartments" : basket_fs_neuron.copy()
        }
        neuron["compartments"]["V0"] += 10 * np.random.rand()
        neuron["compartments"]["Iextmean"] += 0.5*np.random.randn()
        neurons.append(neuron)
             
    for idx in range(Nolm):
        neuron = {
            "type" : "olm_cell",
            "compartments" : olm_params.copy()
        }
        neuron["compartments"]["V0"] += 10 * np.random.rand()
        neuron["compartments"]["Iextmean"] += 0.5*np.random.randn()
        neurons.append(neuron)
    
    # set synapses intro hippocampal model
    for idx in range(Ns):
        pre_ind = np.random.randint(0, Np) + NNseptum
        post_ind = np.random.randint(0, Np) + NNseptum
        if (pre_ind == post_ind):
            post_ind = np.random.randint(0, Np) + NNseptum
        synapse = {
            "type" : "ComplexSynapse",
            "pre_ind": pre_ind, 
            "post_ind": post_ind,
            "pre_compartment_name": "soma",
            "post_compartment_name" : "soma",
            "params" : ext_synapse_params.copy(),
        }
        synapse["params"]["delay"] = np.random.randint(20, 50)
        synapses.append(synapse)
        
    for _ in range(Nint2pyr):
        for idx in range(Np):
            pre_ind = np.random.randint(Np, Np + Nb) + NNseptum
            synapse = {
                "type" : "ComplexSynapse",
                "pre_ind": pre_ind, 
                "post_ind": idx + NNseptum,
                "pre_compartment_name": "soma",
                "post_compartment_name" : "soma",
                "params": inh_synapse_params.copy(),
            }
            synapse["params"]["delay"] = np.random.randint(20, 50)
            synapses.append(synapse)
        
    for _ in range(Nint2pyr):
        for idx in range(Np):
            pre_ind = np.random.randint(Np + Nb, Np + Nb + Nolm) + NNseptum
            synapse = {
                "type" : "ComplexSynapse",
                "pre_ind": pre_ind, 
                "post_ind": idx + NNseptum,
                "pre_compartment_name": "soma",
                "post_compartment_name" : "dendrite",
                "params": inh_synapse_params.copy(),
            }
            synapse["params"]["delay"] = np.random.randint(20, 50)
            synapses.append(synapse)
        
        
    for idx in range(Np):
        pre_ind = idx + NNseptum
        # set synapse from pyramide to OLM neuron
        post_ind = np.random.randint(Np + Nb, Np + Nb + Nolm) + NNseptum
        synapse = {
            "type" : "ComplexSynapse",
            "pre_ind": pre_ind, 
            "post_ind": post_ind,
            "pre_compartment_name": "soma",
            "post_compartment_name" : "soma",
            "params": ext_synapse_params.copy(),
        }
        synapse["params"]["delay"] = np.random.randint(20, 50)
        synapses.append(synapse)
            
         # set synapse from pyramide to basket neuron
        post_ind = np.random.randint(Np, Np + Nb) + NNseptum
        synapse = {
            "type" : "ComplexSynapse",
            "pre_ind": pre_ind, 
            "post_ind": post_ind,
            "pre_compartment_name": "soma",
            "post_compartment_name" : "soma",
            "params": ext_synapse_params.copy(),
        }
        synapse["params"]["delay"] = np.random.randint(20, 50)
        synapses.append(synapse)
    
    # set synapses from MSDB PV-neurons to hippocampal interneurons
    # from PV1-neurons to basketcells   
    for idx in range(Ns2in):
        pre_ind = np.random.randint(Nglu+NgabaCR, Nglu+NgabaCR+NgabaPV1+NgabaPacPV1)
        
        post_ind = np.random.randint(Np, Np + Nb) + NNseptum
        
        
    
        synapse = {
            "type" : "ComplexSynapse",
            "pre_ind": pre_ind, 
            "post_ind": post_ind,
            "pre_compartment_name": "soma",
            "post_compartment_name" : "soma",
            "params": inh_synapse_params.copy(),
        }
        # synapse["params"]["Erev"] = -75
        synapse["params"]["w"] = 50
        synapse["params"]["delay"] = np.random.randint(20, 50)
        synapses.append(synapse)
        
    # from PV2-neurons to OLM neurons    
    for idx in range(Ns2in):
        pre_ind = np.random.randint(Nglu+NgabaCR+NgabaPV1+NgabaPacPV1, Nglu+NgabaCR+NgabaPV1+NgabaPacPV1+NgabaPV2+NgabaPacPV2)
        
        
        post_ind = np.random.randint(Np + Nb, Np + Nb + Nolm) + NNseptum
        synapse = {
            "type" : "ComplexSynapse",
            "pre_ind": pre_ind, 
            "post_ind": post_ind,
            "pre_compartment_name": "soma",
            "post_compartment_name" : "soma",
            "params": inh_synapse_params.copy()
         }
         # synapse["params"]["Erev"] = -75
        synapse["params"]["w"] = 50
        synapse["params"]["delay"] = np.random.randint(20, 50)
        synapses.append(synapse)
    
    # set hippocampal synapses to MSBD PV-neurons
    # from OLM cells to PV1-neurons
    
    if (experiment == "OLM2PV1" or experiment == "OLM2PV1andPV2"):
        for idx in range(NsOLM2PV):
            post_ind = np.random.randint(Nglu+NgabaCR, Nglu+NgabaCR+NgabaPV1+NgabaPacPV1)
            pre_ind = np.random.randint(Np + Nb, Np + Nb + Nolm) + NNseptum
        
            synapse = {
                "type" : "ComplexSynapse",
                "pre_ind": pre_ind, 
                "post_ind": post_ind,
                "pre_compartment_name": "soma",
                "post_compartment_name" : "soma",
                "params": inh_synapse_params.copy(),
            }
            # synapse["params"]["Erev"] = -75
            synapse["params"]["w"] = 50
            synapse["params"]["delay"] = np.random.randint(20, 50)
            synapses.append(synapse)

    if (experiment == "OLM2PV2" or experiment == "OLM2PV1andPV2"):
        for idx in range(NsOLM2PV):
            post_ind = np.random.randint(Nglu+NgabaCR+NgabaPV1+NgabaPacPV1, Nglu+NgabaCR+NgabaPV1+NgabaPacPV1+NgabaPV2+NgabaPacPV2)
            pre_ind = np.random.randint(Np + Nb, Np + Nb + Nolm) + NNseptum
        
            synapse = {
                "type" : "ComplexSynapse",
                "pre_ind": pre_ind, 
                "post_ind": post_ind,
                "pre_compartment_name": "soma",
                "post_compartment_name" : "soma",
                "params": inh_synapse_params.copy(),
            }
            # synapse["params"]["Erev"] = -75
            synapse["params"]["w"] = 50
            synapse["params"]["delay"] = np.random.randint(20, 50)
            synapses.append(synapse)
            
            
            
    duration = 500
    dt = 0.1
    sim = SimmulationParams()
    net = lib.Network(neurons, synapses)
    for _ in range(4):
        curent_time = time.time()
        net.integrate(dt, duration, sim.iext_function)
        print ("Calculation time is %0.3f sec" % (time.time() - curent_time) )
        net.save_results(path+"all_results")
        """
        firing = net.getFiring()
        np.save(path + "firing", firing)
        
        plt.figure()
        plt.plot(firing[0, :], firing[1, :], ".")
        plt.ylim(0, 680)
        plt.show()
        
        lfp = net.getLFP()
        np.save(path + "lfp", lfp)
        
        t = np.linspace(0, lfp.size*dt, lfp.size)
        plt.figure()
        plt.plot(t, lfp)
        plt.show()
        
        V = net.getVhist()
        np.save(path + "V", V)
        currents = net.getfullLFP()
        np.save(path + "currents", currents)
        """
        
#####################################################
path = "/home/ivan/Data/modeling_septo_hippocampal_model/septo_hippocampal_model2/"
experiments = ["without_hipp-septum_input", "OLM2PV1", "OLM2PV2", "OLM2PV1", "OLM2PV1andPV2"]
for experiment in experiments:
    for idx in range(5):
        tmp_path = path + experiment + "/"
        if not (os.path.isdir(tmp_path)):
            os.mkdir(tmp_path)
        tmp_path += str(idx + 1)
        
        run_model(tmp_path, experiment)



