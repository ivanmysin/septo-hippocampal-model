#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fast implementation of HH equations
"""
from libc.math cimport  cos, exp
from libcpp.map cimport map
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
from cython.operator cimport dereference, preincrement
import numpy as np
cimport numpy as np
from libcpp.queue cimport queue
from cython.parallel cimport parallel, prange
cimport cython

# exp = np.exp

# base class for currents
cdef class Current:
    cdef double I
    
    def __cinit__ (self, params):
        pass
        
    cpdef void update_gate_vars(self, double V, double dt):
        pass
              
    cpdef void init_gate_vars(self, double V):
        pass
    
    cpdef void precumpute_funtions(self, double Vmin, double Vmax, double step, double dt):
        pass
    
    cpdef void update_gate_vars_fast(self, double V, double dt):
        pass
    
    cpdef double getI(self, double V):
        return self.I


cdef class ExternalCurrent(Current):
    cdef double varience
    def __cinit__ (self, params):
        self.I = -params["Iext"]
        self.varience = params["varience"]
        
    cpdef double getI(self, double V):
        if (self.varience > 0):
            return np.random.normal(self.I, self.varience)
        else:
            return self.I
        

# class for leak current
cdef class LeakCurrent(Current):
    cdef double Erev, gmax
    
    def __cinit__(self, params):
        self.Erev = params["Erev"] # save reverse potential
        self.gmax = params["gmax"] # save maximal conductance
    
   
    cpdef double getI(self, double V):
        """
        return current value
        """
        return self.gmax * (V - self.Erev)

cdef class SimpleSynapse:
    cdef double Erev, gmax, w
    cdef double s, tau
    cdef OriginCompartment pre, post
    
    def __cinit__(self, params):
        self.Erev = params["Erev"] # save reverse potential
        self.gmax = params["gmax"] # save maximal conductance
        self.w = params["w"]
        self.tau = params["tau"]
        self.pre = params["pre"]
        self.post = params["post"]
        self.s = 0
    
    cpdef void update_gate_vars(self, double V, double dt):
        if ( self.pre.get_fired() ):
            self.s = 1
            return
        if (self.s < 0.001):
            self.s = 0
            return
        
        cdef double Vpost = self.post.getV()
        cdef double Isyn = self.w * self.gmax * self.s * (Vpost - self.Erev)
        self.post.addIsyn(Isyn) #  Isyn for post neuron
        
        cdef double k1 = self.s
        cdef double k2 = k1 - 0.5 * dt * (k1 / self.tau)
        cdef double k3 = k2 - 0.5 * dt * (k2 / self.tau)
        cdef double k4 = k1 - dt * (k1 / self.tau)
        
        self.s = (k1 + 2*k2 + 2*k3 + k4) / 6.0
    
    cpdef double getI(self, double V):
        """
        return current value
        """
        return self.gmax * self.s * (V - self.Erev)


    
# class for sodium current
cdef class SodiumCurrent4FS(LeakCurrent):
    # save x_inf and T functions in class atributes
    cdef vector [double] m_inf_fast, h_inf_fast, h_T_fast
    # parameters for precomputed functions
    cdef double Vmin # = 0
    cdef double Vmax # = 0
    cdef double step # = 0
     
    cdef double fi
    cdef double m, h

    

    
    def __cinit__(self, params):

        self.fi = params["fi"] # save threshold of spike genereation
        # declare gate variables
        self.m = 0
        self.h = 0
    
    cpdef void init_gate_vars(self, double V):
        
        """
        initiate gate variables before run of simulation,
        m = m_inf and h = h_inf
        """
        cdef double alpha_m = self.get_alpha_m(V)
        cdef double beta_m = self.get_beta_m(V)
        self.m = alpha_m / (alpha_m + beta_m)
        
        cdef double alpha_h = self.get_alpha_h(V)
        cdef double beta_h = self.get_beta_h(V)
        self.h = alpha_h / (alpha_h + beta_h)
    
    
    cpdef void update_gate_vars(self, double V, double dt):
        """
        update values of gate variables for new V on time step dt
        """
        cdef double alpha_m = self.get_alpha_m(V)
        cdef double beta_m = self.get_beta_m(V)
        self.m = alpha_m / (alpha_m + beta_m)
        
        cdef double alpha_h = self.get_alpha_h(V)
        cdef double beta_h = self.get_beta_h(V)
        
        cdef double tau_inf = 1 / (alpha_h + beta_h)
        cdef double h_inf = alpha_h * tau_inf
        
        self.h = h_inf - (h_inf - self.h) * exp(-dt/tau_inf)
    
    
    cpdef void precumpute_funtions(self, double Vmin, double Vmax, double step, double dt):
        """
        precompute x_inf and T functions from Vmin to Vmax with step, dt need precomputing T
        we save x_inf and T functions and their parameters to class atributes
        
        """
#        if ( self.__class__.step > 0 ):
#            return
        
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.step = step
        
        cdef int size = np.ceil ( (self.Vmax - self.Vmin) / self.step )
        
        
        self.m_inf_fast = vector[double](size)
        self.h_inf_fast = vector[double](size)
        self.h_T_fast = vector[double](size)
        
        cdef double V = Vmin
        cdef double tau_h
        cdef int idx = 0
        while (idx < size):
            self.m_inf_fast[idx] =  self.get_alpha_m(V) / (self.get_alpha_m(V) + self.get_beta_m(V)) 
            
            tau_h = 1 / (self.get_alpha_h(V) + self.get_beta_h(V))
            
            self.h_inf_fast[idx] = self.get_alpha_h(V) * tau_h
    
            self.h_T_fast[idx] = exp(-dt / tau_h)
            
            V += self.step
            idx += 1
        
    
    cpdef void update_gate_vars_fast(self, double V, double dt):
        """
        update values of gate variables for new V with usage of precomputed values
        """
        
        idx = int( (V - self.Vmin) / self.step ) # index for precomputed array

        self.m = self.m_inf_fast[idx]

        h_inf = self.h_inf_fast[idx]
        T_h = self.h_T_fast[idx]
        
        self.h = h_inf - (h_inf - self.h) * T_h
    
    
    
    cpdef double getI(self, double V):
        return self.gmax * self.m * self.m * self.m * self.h * (V - self.Erev)
    
    cdef double get_alpha_m(self, double V):
         cdef double  x = -0.1 * (V + 33)
         if (x == 0):
            x = 0.000000001
         cdef double alpha = x / ( exp(x) - 1 )
         return alpha
#########
    cdef double get_beta_m(self, double V):
        cdef double beta = 4 * exp(- (V + 58) / 18 )
        return beta


########
    cdef double get_alpha_h(self, double V):
        cdef double alpha = self.fi * 0.07 * exp( -(V + 51) / 10)
        return alpha

########
    cdef double get_beta_h(self, double V):
        cdef double beta = self.fi / ( exp(-0.1 * (V + 21)) + 1 )
        return beta
    
########
    cdef double get_alpha_n(self, double V):
        cdef double x = -0.1 * (V + 38)
        if ( x==0 ):
            x = 0.00000000001
        cdef double alpha = self.fi * 0.1 * x / (exp(x) - 1)
        return alpha
#######np.

    cdef double get_beta_n(self, double V):
        return (self.fi * 0.125 * exp( -(V + 48 )/ 80))


    
cdef class PotassiumCurrent4FS(LeakCurrent):

    cdef vector[double] n_inf_fast, n_T_fast
    
    cdef double Vmin # = 0.0
    cdef double Vmax # = 0.0
    cdef double step # = 0.0
    
    
    
    cdef double n, fi
    
    def __cinit__(self, params):

        self.fi = params["fi"]
        self.n =  0

    
    cpdef void update_gate_vars(self, double V, double dt):
        """
        update n
        """
        cdef double alpha_n = self.get_alpha_n(V)
        cdef double beta_n = self.get_beta_n(V)
        cdef double tau_inf = 1 / (alpha_n + beta_n)
        cdef double  n_inf = alpha_n * tau_inf
        
        self.n = n_inf - (n_inf - self.n) * exp(-dt/tau_inf)
    
    cpdef void init_gate_vars(self, double V):
        """
        initialize n = n_inf
        """
        cdef double alpha_n = self.get_alpha_n(V)
        cdef double beta_n = self.get_beta_n(V)
        self.n = alpha_n / (alpha_n + beta_n)
    
    cpdef double getI(self, double V):
        return self.gmax * self.n * self.n * self.n * self.n * (V - self.Erev) 



    cpdef void precumpute_funtions(self, double Vmin, double Vmax, double step, double dt):
        """
        precompute x_inf and T functions from Vmin to Vmax with step, dt need precomputing T
        we save x_inf and T functions to class atributes
        
        """        
#        if ( self.__class__.step > 0 ):
#            return
        
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.step = step
        cdef int size = np.ceil ( (self.Vmax - self.Vmin) / self.step )

        self.n_inf_fast = vector[double](size)
        self.n_T_fast = vector[double](size)
        
        cdef double V = self.Vmin
        cdef int idx = 0
        while ( idx < size ):
            tau_n = 1 / (self.get_alpha_n(V) + self.get_beta_n(V))
            
            self.n_inf_fast[idx] = self.get_alpha_n(V) * tau_n 
            self.n_T_fast[idx] = exp(-dt / tau_n)
            
            V += self.step
            idx += 1

    cpdef void update_gate_vars_fast(self, double V, double dt):
        """
        update values of gate variables for new V with usage of precomputed values
        """        
        cdef int idx = int ( (V - self.Vmin) / self.step )  # index for precomputed array
        if (idx < 0):
            print (idx, V)

        cdef double n_inf = self.n_inf_fast[idx]
        cdef double T_n = self.n_T_fast[idx]
        
        self.n = n_inf - (n_inf - self.n) * T_n

    cdef double get_alpha_n(self, double V):
        cdef double x = -0.1 * (V + 38)
        if ( x == 0 ):
            x = 0.00000000001
        cdef double alpha = self.fi * 0.1 * x / (exp(x) - 1)
        return alpha

    cdef double get_beta_n(self, double V):
        return (self.fi * 0.125 * exp( -(V + 48 )/ 80))
 
   
    
    
    
cdef class OriginCompartment:
    cdef  double V, Isyn
    
    cdef np.ndarray Vhist
    cdef np.ndarray firing
    
    def __cinit__(self, params):
        self.Isyn = 0
    
    cdef double getV(self):
        return self.V

   
    def getVhist(self):
        return self.Vhist
    
   
    def getFiring(self):
        return self.firing
        
    cpdef integrate(self, double dt, double duration):
        pass
    
    def getCompartmentsNames(self):
        return ["soma"]
 
    cpdef checkFired(self, double t_):
       pass
   
    cpdef addIsyn(self, double Isyn):
        self.Isyn += Isyn

cdef class Compartment(OriginCompartment):
    # cdef Current currents
    cdef np.ndarray currents
    
    # cdef vector[Current] cur
    
    cdef int n_currents
    cdef double C
    def __cinit__(self, params):
        
        self.V = params["V0"]
        self.C = params["C"]
        self.Vhist = np.array([self.V]) # array for save history of V 
        
        self.n_currents = params["n_currents"]
        # self.currents = <Current> [self.n_currents]
        self.currents = np.empty((0), dtype=object)

        
        # self.cur = vector[Current]()
        
    cpdef addCurrent(self, current):
        """
        add current to neuron model
        """
       

#        if (current_params["type"] == "external"):
#            current = ExternalCurrent(current_params)
#
#        if (current_params["type"] == "leak"):
#            current = LeakCurrent(current_params)
#
#        if (current_params["type"] == "sodium_current4FS"):
#            current = SodiumCurrent4FS(current_params)
#
#        if (current_params["type"] == "potassium_current4FS"):
#            current = PotassiumCurrent4FS(current_params)

        
        current.init_gate_vars(self.V)
        self.currents = np.append(self.currents, current)


    
    cdef void updateV(self, double dt):
        """
        update V for time step dt
        Euler methods is used 
        """
        cdef double I = 0
        self.n_currents = self.currents.size
        for i in range(self.n_currents):

            self.currents[i].update_gate_vars(self.V, dt)

            I -= self.currents[i].getI(self.V)
        I -= self.Isyn
        self.V += dt * (I / self.C)
        
        self.Vhist = np.append(self.Vhist, self.V)
        self.Isyn = 0

        
    cdef void updateVfast(self, double dt):
        """
        update V for time step dt with usage updateIfast methods of currents
        Euler methods is used 
        """
        cdef double I = 0
        
        self.n_currents = self.currents.size
        for i in range(self.n_currents):
                
            self.currents[i].update_gate_vars_fast(self.V, dt)
            I -= self.currents[i].getI(self.V)
            
        
        self.V += dt * (I / self.C)
        
        self.Vhist = np.append(self.Vhist, self.V)
    
    cpdef run(self, double dt=0.01, double duration=200):
        """
        run simulation of neuron for duration with time step dt
        """

        cdef double t = 0
        while(t <= duration):
            self.updateV(dt)
            
            t += dt
            
            
    cpdef void runfast(self, double dt=0.01, double duration=200):
        """
        run simulation of neuron for duration with time step dt with usage of precomputed functions
        """
        cdef double Vmin = -100.0
        cdef double Vmax = 70.0
        cdef double step = 0.01
        
        for i in range(self.n_currents):
            self.currents[i].precumpute_funtions(Vmin, Vmax, step, dt)

        cdef double t = 0
        while(t <= duration):
            
            self.updateVfast(dt)
            t += dt

cdef class Network:
    cdef np.ndarray neurons
    cdef np.ndarray synapses
    cdef list neuron_params, synapse_params
    cdef double t
    def __cinit__(self, neuron_params, synapse_params):
        self.neuron_params = neuron_params
        self.synapse_params = synapse_params
        self.neurons = np.empty((0), dtype=object)
        self.synapses = np.empty((0), dtype=object)
        self.t = 0

    cpdef void addNeuron(self,neuron):
        self.neurons = np.append(self.neurons, neuron)
    
    cdef void addSynapse(self, synapse):
        self.synapses = np.append(self.synapses, synapse)
           
    cpdef run(self, double dt, double duration):
        
        cdef double Iext_model
        cdef int NN = self.neurons.size
        cdef int NS = self.synapses.size
        cdef int s_ind = 0
        cdef int neuron_ind = 0
        cdef double t = 0
        while(t < duration):
            for neuron_ind in range(NN):
                self.neurons[neuron_ind].run(dt, dt)    
                    
            for s_ind in range(NS):
                self.synapses[s_ind].init_gate_vars(dt)
                

            t += dt
            self.t += dt
            
#    def getVhist(self):
#        V = []
#        for idx, n in enumerate(self.neurons):
#            Vn = dict()
#            for key in n.getCompartmentsNames():
#                Vn[key] = n.getCompartmentByName(key).getVhist()
#            V.append(Vn)
#        return V
#
#    def getLFP(self, layer_name="soma"):
#        lfp = 0
#        for idx, n in enumerate(self.neurons):
#            # lfp += n.getCompartmentByName("soma").getLFP()
#            
#            
#            for key in n.getCompartmentsNames():
#                # Vn[key] = n.getCompartmentByName(key).getVhist()
#                Vext =  n.getCompartmentByName(key).getLFP()
#                if (key == "soma"):
#                    lfp -= Vext
#                else:
#                    lfp -= Vext / 10
#
#        return lfp
#
#    def getFiring(self):
#        firing = np.empty((2, 0), dtype=np.float64)
#        for idx, n in enumerate(self.neurons):
#            fired = n.getCompartmentByName("soma").getFiring()
#            fired_n = np.zeros_like(fired) + idx + 1
#            firing = np.append(firing, [fired, fired_n], axis=1)
#            
#            
#        return firing
       
#    def save_results(self, file):
#        result = {
#            "simulation_params" : {},
#            "results" : {},
#        }
#        result["simulation_params"]["neurons"] = self.neuron_params
#        result["simulation_params"]["synapses"] = self.synapse_params
#        
#        result["results"]["lfp"] = self.getLFP()
#        result["results"]["firing"] = self.getFiring()
#        result["results"]["V"] = self.getVhist()
#        result["results"]["currents"] = self.getfullLFP()
#        
#        np.save(file, result)
