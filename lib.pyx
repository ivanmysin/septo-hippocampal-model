# -*- coding: utf-8 -*-
"""
class lib for modeling
"""
import numpy as np
cimport numpy as np
from libc.math cimport exp

cdef class FS_neuron:
    cdef double V, ENa, EK, El, gbarNa, gbarK, gl, fi, Iextmean, Iextvariance
    cdef double Iext, m, n, h, gNa, gK, Isyn, th
    cdef int countSp
    cdef np.ndarray firing
    
    #cdef public void integrate(self, double, double)
    # cdef public void checkFired(self, double)
    cdef public double getV(self)        
    cdef public void setIsyn(self, double)
    cdef public void setIext(self, double)
    # cdef public np.ndarray getFiring(self)
       
    cdef double alpha_m(self)
    cdef double beta_m(self)
    cdef double alpha_h(self)
    cdef double beta_h(self)
    cdef double alpha_n(self)
    cdef double beta_n(self)
    cdef double h_integrate(self, double)
    cdef double n_integrate(self, double)

    
    def __cinit__(self, params):
         self.V = params[0]
         self.ENa = params[1] # mv Na reversal potential
         self.EK  = params[2] # mv K reversal potential
         self.El  = params[3] # mv Leakage reversal potential
         self.gbarNa = params[4] # mS/cm^2 Na conductance
         self.gbarK  = params[5]  # mS/cm^2 K conductance
         self.gl = params[6]
         self.fi = params[7]
         self.Iextmean = params[8]
         self.Iextvariance = params[9]
         
         self.Iext = np.random.normal(self.Iextmean, self.Iextvariance) 
         
         # self.Vhist = []
         self.firing = np.empty(0, dtype=np.float64)
         
         self.m = self.alpha_m() / (self.alpha_m() + self.beta_m())
         self.n = self.alpha_n() / (self.alpha_n() + self.beta_n())
         self.h = self.alpha_h() / (self.alpha_h() + self.beta_h())

         self.gNa = self.gbarNa * self.m * self.m * self.m * self.h
         self.gK = self.gbarK * self.n * self.n * self.n * self.n

         self.Isyn = 0
         self.countSp = True
         self.th = -20


    cdef double alpha_m(self):
            #double alpha;
         cdef double x = -0.1 * (self.V + 33)
         if (x == 0):
            x = 0.000000001
         cdef double alpha = x / ( exp(x) - 1 )
         return alpha
#########
    cdef double beta_m(self):
        cdef double beta = 4 * exp(- (self.V + 58) / 18 )
        return beta


########
    cdef double alpha_h(self):
        cdef double alpha = self.fi * 0.07 * exp( -(self.V + 51) / 10)
        return alpha

########
    cdef double beta_h(self):
        cdef double beta = self.fi / ( exp(-0.1 * (self.V + 21)) + 1 )
        return beta
    
########
    cdef double alpha_n(self):
        cdef double x = -0.1 * (self.V + 38)
        if ( x==0 ):
            x = 0.00000000001
        
        cdef double alpha = self.fi * 0.1 * x / (exp(x) - 1)
        return alpha
#######

    cdef double beta_n(self):
        return (self.fi * 0.125 * exp( -(self.V + 48 )/ 80))

#######
    cdef double h_integrate(self, double dt):
    
        cdef double h_0 = self.alpha_h() / (self.alpha_h() + self.beta_h())
        cdef double tau_h = 1 / (self.alpha_h() + self.beta_h())
        return h_0 -(h_0 - self.h) * exp(-dt/tau_h)
#######

    cdef double n_integrate(self, double dt):

        cdef double n_0 = self.alpha_n() / (self.alpha_n() + self.beta_n() )
        cdef double tau_n = 1 / (self.alpha_n() + self.beta_n())
        return n_0 -(n_0 - self.n) * exp(-dt/tau_n)

#######
    def integrate (self, double dt, double duraction):

        cdef double t = 0
        cdef int i = 0
        while (t < duraction):
            # self.Vhist.append(self.V)
            
            self.V = self.V + dt * (self.gNa * (self.ENa - self.V) + self.gK * (self.EK - self.V) + self.gl*(self.El - self.V) - self.Isyn + self.Iext)
    
            self.m = self.alpha_m() / (self.alpha_m() + self.beta_m())
            self.n = self.n_integrate(dt)
            self.h = self.h_integrate(dt)
            
            self.gNa = self.gbarNa * self.m * self.m * self.m * self.h
            self.gK = self.gbarK * self.n * self.n * self.n * self.n
    
            self.Iext = np.random.normal(self.Iextmean, self.Iextvariance) 
    
            self.Isyn = 0
            i += 1
            t += dt
    

########
    def checkFired(self, double t_):
    
        if (self.V >= self.th and self.countSp):
            self.firing = np.append(self.firing, np.array([t_]), axis=0)
            self.countSp = False
        
        if (self.V < self.th):
            self.countSp = True
######## 
    cdef double getV(self):
        return self.V
        
##    def getVhist(self):
##        return self.Vhist
        
    cdef void setIsyn(self, double Isyn):
        self.Isyn += Isyn
    
    cdef void setIext(self, double Iext):
        self.Iext = Iext
    
    def getFiring(self):
        return self.firing
    
######################################################
cdef class Pac_neuron(FS_neuron):

    cdef double Eh, gbarKS, gbarH, H, p, q, gKS, gH 
    # cdef public void integrate(self, double, double)
    cdef double H_integrate(self, double)
    cdef double p_integrate(self, double)
    cdef double q_integrate(self, double)
    
    def __cinit__(self, params):
        super().__init__(params)
        
        self.Eh = params[10]
        self.gbarKS = params[11]
        self.gbarH = params[12]

        self.H = 1 / (1 + exp( (self.V + 80 ) / 10) )
        self.p = 1 / (1 + exp(-(self.V + 34) / 6.5) )
        self.q = 1 / (1 + exp( (self.V + 65) / 6.6) )

        self.gKS = self.gbarKS * self.p * self.q
        self.gH = self.gbarH * self.H


    cdef double H_integrate(self, double dt):
    
        cdef double H_0 = 1 / (1 + exp( (self.V + 80) / 10) )
        cdef double tau_H = ( 200 / (exp( (self.V + 70) / 20) + exp( -(self.V + 70) / 20) ) ) + 5
        return H_0 - (H_0 - self.H) * exp(-dt/tau_H)

######
    cdef double p_integrate(self, double dt):
    
        cdef double p_0 = 1 / ( 1 + exp( -(self.V + 34) / 6.5 ) )
        cdef double tau_p = 6
        return p_0 - (p_0 - self.p) * exp(-dt/tau_p)

######
    cdef double q_integrate(self, double dt):
    
        cdef double q_0 = 1 / ( 1 + exp( (self.V + 65) / 6.6) )
        cdef double tau_q0 = 100
        cdef double tau_q = tau_q0 * ( 1 + ( 1 / ( 1 + exp( -(self.V + 50) / 6.8) ) ) )
        return q_0 - (q_0 - self.q) * exp(-dt/tau_q)
    
    def integrate (self, double dt, double duraction):
    
        cdef double t = 0
        cdef int i = 0
        while (t < duraction):
            # self.Vhist.append(self.V)
            self.V = self.V + dt*(-self.gNa*(self.V - self.ENa) - self.gK*(self.V - self.EK) - self.gl*(self.V - self.El) - self.gKS*(self.V - self.EK) - self.gH*(self.V - self.Eh) - self.Isyn + self.Iext)
    
            self.m = self.alpha_m() / (self.alpha_m() + self.beta_m())
            self.n = self.n_integrate(dt)
            self.h = self.h_integrate(dt)
            self.H = self.H_integrate(dt)
            self.p = self.p_integrate(dt)
            self.q = self.q_integrate(dt)
    
            self.gNa = self.gbarNa * self.m * self.m * self.m * self.h
            self.gK = self.gbarK * self.n * self.n * self.n * self.n
    
            self.gKS = self.gbarKS * self.p * self.q
            self.gH = self.gbarH * self.H
            self.Isyn = 0
    
            self.Iext = np.random.normal(self.Iextmean, self.Iextvariance) 
            
            i += 1
            t += dt
        

cdef class Synapse:
    
    cdef FS_neuron pre, post
    cdef double Erev, gbarS, w, alpha_s, beta_s, teta, K, S, Isyn 
    
    def __cinit__ (self, pre, post, params):
    
        self.pre = pre
        self.post = post
        self.Erev = params[2]
        self.gbarS = params[3]
        self.w = params[4]
        self.alpha_s = params[5]
        self.beta_s = params[6]
        self.teta = params[7]
        self.K = params[8]
        self.S = 0
        self.Isyn = 0


    def integrate(self, double dt, double duraction):
    
        cdef double Vpre = self.pre.getV() # V of pre neuron
        if (Vpre < -30 and self.S<0.005):
            self.S = 0
            return
    
        cdef double Vpost = self.post.getV()
        cdef double F = 1 / (1 + exp( -( Vpre - self.teta)/ self.K) )
        cdef double S_0 = self.alpha_s * F / (self.alpha_s * F + self.beta_s)
        cdef double tau_s = 1 / (self.alpha_s * F + self.beta_s)
        self.S = S_0 - (S_0 - self.S) * exp( -duraction / tau_s )
        self.Isyn = self.w * self.gbarS * self.S * (Vpost - self.Erev)
        self.post.setIsyn(self.Isyn) #  Isyn for post neuron

cdef class Network:
    cdef int n, numberSynapses
    cdef list neurons
    cdef list synapses
    def __init__(self, neurons_properties, synapses_propeties):

        self.n = len(neurons_properties)
        self.neurons = []
        for i in range(self.n):
            if (len(neurons_properties[i])<12):
                self.neurons.append(FS_neuron(neurons_properties[i]))
            else:
                self.neurons.append(Pac_neuron(neurons_properties[i]))
        
    

        self.numberSynapses = len(synapses_propeties) # set array of synapses
        self.synapses = []
        for i in range(self.numberSynapses):
            pre_ind = int(synapses_propeties[i][0])
            post_ind = int(synapses_propeties[i][1])
            self.synapses.append(Synapse (self.neurons[pre_ind], self.neurons[post_ind], synapses_propeties[i]))
    



#############
    """
    def __del__(self):
    
        for i in range(self.n):
            del self.neurons[i]
        
        del self.neurons
        
        for i in range(self.numberSynapses): 
            del self.synapses[i]
        
        del self.synapses
    """
############
# intergrate model
    def integrate (self, double dt, double duraction):
    
    
        cdef double t = 0
        while (t < duraction): 
            for i in range(self.n): # integrate all neurons by one time step
                self.neurons[i].integrate(dt, dt)
                self.neurons[i].checkFired(t)
            
    
            for i in range(self.numberSynapses): # integrate all synapses by one time step
                self.synapses[i].integrate(dt, dt)
            
    
            t += dt


#########
# save raster of activity in csv file
    def saveRaster (self, filepath):
    
        
        outfile = open(filepath)
        for i in range(self.n): 
            sp = self.neurons[i].getFiring()
        
            for j in range(len(sp)):
                outfile.write( str(i+1) +  ", " + str(sp[j]) + "\n" )
            
        
    
        outfile.close()

##########
# return times in sec of spikes of neuron by its index
    def getContiniousFiring (self, int neuronInd, double dt, double duraction):
    
        cdef int length = int (duraction / dt)
        cdef np.ndarray firing = self.neurons[neuronInd].getFiring()
        cdef np.ndarray signal = np.zeros(length)
        signal[(firing/dt).astype(np.int)] = 1
        return signal

    def getRaster(self):
        cdef np.ndarray firing = np.empty(shape=[0, 2], dtype=float)
        for i in range(self.n): 
            sp = self.neurons[i].getFiring()
            if (sp.size > 0):
                sp = sp.reshape(sp.size, 1)
                nn = np.zeros([sp.size, 1])+i+1
                sp = np.append(sp, nn, axis=1)            
                firing = np.append(firing, sp, axis=0)
        return firing


        
