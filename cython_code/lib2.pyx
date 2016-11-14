# -*- coding: utf-8 -*-
"""
lib full cython 
"""
from libc.math cimport exp, cos
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


cdef class OriginCompartment:
    cdef  double V, Isyn, Iext, Icoms
    cdef np.ndarray Vhist
    cdef np.ndarray LFP
    cdef np.ndarray firing
    
    def __cinit__(self, params):
        pass
    
    cdef double getV(self):
        return self.V
        
    
    cdef void setIext(self, double Iext):
        self.Iext = Iext
        
    def addIsyn(self, double Isyn):
        self.Isyn += Isyn
    
    def addIcoms(self, double Icoms):
        self.Icoms += Icoms  
    
        
    def getVhist(self):
        return self.Vhist
    
    def getLFP(self):
        return self.LFP
    
    def getFiring(self):
        return self.firing
        
    cpdef integrate(self, double dt, double duration):
        pass
    
    def getCompartmentsNames(self):
        return ["soma"]

    def getCompartmentByName(self, name): 
        return self
 
    cpdef checkFired(self, double t_):
       pass

cdef class PoisonSpikeGenerator(OriginCompartment):
    cdef double t, latency, lat, probability
    
    def __cinit__(self, params):
        self.t = 0
        self.latency = params["latency"] # in ms
        self.probability = params["probability"] # probability of spike generation in time moment
        self.firing = np.array([])
        self.lat = -self.latency/2
    def integrate(self, double dt, double duration):
        self.V = -60
        self.lat -= dt
        
        cdef double tmp = np.random.rand()
        if (self.lat <= 0 and self.probability > tmp):
            self.V = 50
            self.lat = self.latency
            self.firing = np.append(self.firing, 1000 * self.t)

        self.t += 0.001 * dt
        
    def getVhist(self):
        return 0
        
    def getLFP(self):
        return np.zeros(1)
    
    def getFiring(self):
        return self.firing

cdef class CosSpikeGenerator(OriginCompartment):
    cdef double t, freq, phase, latency, lat, probability
    
    def __cinit__(self, params):
        self.t = 0
        self.freq = params["freq"] # frequency in Hz
        self.phase = params["phase"]
        self.latency = params["latency"] # in ms
        self.probability = params["probability"] # probability of spike generation in time moment
        self.firing = np.array([])
        self.lat = -self.latency/2
        
    def integrate(self, double dt, double duration):

        self.V = -60
        self.lat -= dt
        
        cdef double signal = cos(2 * np.pi * self.t * self.freq + self.phase)
        if (signal > 0.5 and self.lat <= 0 and self.probability > np.random.rand() ):
            self.V = 50
            self.lat = self.latency
            self.firing = np.append(self.firing, 1000 * self.t)

        self.t += 0.001 * dt
    
    def getVhist(self):
        return 0
        
    def getLFP(self):
        return np.zeros(1)
    
    def getFiring(self):
        return self.firing
        
cdef class OLM_cell(OriginCompartment):
    cdef double Capacity, Iextmean, Iextvarience, ENa, EK, El, EH
    cdef double gbarNa, gbarK, gl, gbarKa, gbarH
    cdef double th
    cdef bool countSp
    cdef double m, h, n, a, b, r
    cdef double gNa, gK, gKa, gH
    cdef double distance

    def __cinit__(self, params):
         self.V = params["V0"]
         self.Iextmean = params["Iextmean"]        
         self.Iextvarience = params["Iextvarience"]
         self.ENa = params["ENa"]
         self.EK = params["EK"]
         self.El = params["El"]
         self.EH = params["EH"]
         
         self.gbarNa = params["gbarNa"]
         self.gbarK = params["gbarK"]
         self.gl = params["gl"]   
         self.gbarKa = params["gbarKa"]
         self.gbarH = params["gbarH"]         
         
         self.Iext = np.random.normal(self.Iextmean, self.Iextvarience) 
         
         self.Vhist = np.array([])
         self.LFP = np.array([])
         self.distance = np.random.normal(200, 10)
         self.firing = np.array([])
         
         self.m = self.alpha_m() / (self.alpha_m() + self.beta_m())
         self.n = self.alpha_n() / (self.alpha_n() + self.beta_n())
         self.h = self.alpha_h() / (self.alpha_h() + self.beta_h())
         self.b = self.alpha_b() / (self.alpha_b() + self.beta_b())
         self.a = self.a_inf()
         self.r = self.r_inf()
         
         
         self.gNa = self.gbarNa * self.m * self.m * self.m * self.h
         self.gK = self.gbarK * self.n * self.n * self.n * self.n
         self.gKa = self.gbarKa * self.a * self.b
         self.gH = self.gbarH * self.r 

         self.Isyn = 0
         self.countSp = True
         self.th = -20
    
    cdef double getV(self):
        return self.V + 60

    def getLFP(self):
        return 0

    cdef double alpha_m(self):
         cdef double  x = -0.1 * (self.V + 38)
         if (x == 0):
            x = 0.000000001
         cdef double alpha = x / ( exp(x) - 1 )
         return alpha
#########
    cdef double beta_m(self):
        cdef double beta = 4 * exp(- (self.V + 63) / 20 )
        return beta


########
    cdef double alpha_h(self):
        cdef double alpha = 0.07 * exp( -(self.V + 63) / 20)
        return alpha

########
    cdef double beta_h(self):
        cdef double beta = 1 / ( exp(-0.1 * (self.V + 33)) + 1 )
        return beta
    
########
    cdef double alpha_n(self):
        cdef double x =  (self.V - 25)
        if ( x==0 ):
            x = 0.00000000001
        cdef double alpha = -0.018 * x / (exp(-x/25) - 1)
        return alpha
#######np.

    cdef double beta_n(self):
        cdef double x = self.V - 35
        if ( x==0 ):
            x = 0.00000000001
        
        cdef double beta = 0.0036 * x / (exp(x/12) - 1)
        return beta
    
    cdef double alpha_b(self):
        cdef double alpha = 0.000009 / exp((self.V - 26)/18.5)
        return alpha

    cdef double beta_b(self):
        cdef double beta = 0.014 / (0.2 + exp((self.V + 70)/-11) )
        return beta
    
    cdef double a_inf(self):
        cdef double x = -(self.V + 14) / 16.6
        cdef double a_inf = 1 / (1 + exp(x))
    
    cdef double a_tau(self):
        return 5
        
    cdef double r_inf(self):
        cdef double r_inf = 1 / (1 + exp( (self.V + 84)/10.2 ) )
        return r_inf
    
    cdef double r_tau(self):
        cdef double r_tau = 1 / ( exp(-17.9 - 0.116*self.V) + exp(-1.84 + 0.09*self.V) ) + 100
        return r_tau
#######
    cdef double r_integrate(self, double dt):
        cdef double r_0 = self.r_inf()
        cdef double tau_r = self.r_tau()
        return r_0 - (r_0 - self.r) * exp(-dt/tau_r)
        
    cdef double a_integrate(self, double dt):
        cdef double a_0 = self.a_inf()
        return a_0 - (a_0 - self.a) * exp( -dt/self.a_tau() )
    
    cdef double b_integrate(self, double dt):
        cdef double b_0 = self.alpha_b() / (self.alpha_b() + self.beta_b())
        cdef double tau_b = 1 / (self.alpha_b() + self.beta_b())
        return b_0 - (b_0 - self.b) * exp(-dt/tau_b)
    
    cdef double h_integrate(self, double dt):
        cdef double h_0 = self.alpha_h() / (self.alpha_h() + self.beta_h())
        cdef double tau_h = 1 / (self.alpha_h() + self.beta_h())
        return h_0 - (h_0 - self.h) * exp(-dt/tau_h)
#######
    cdef double n_integrate(self, double dt):
        cdef double n_0 = self.alpha_n() / (self.alpha_n() + self.beta_n() )
        cdef double tau_n = 1 / (self.alpha_n() + self.beta_n())
        return n_0 - (n_0 - self.n) * exp(-dt/tau_n)
#######
    def integrate (self, double dt, double duraction):

        cdef double t = 0
        cdef double i = 0
        while (t < duraction):
            self.Vhist = np.append(self.Vhist, self.V)
            
            self.V = self.V + dt * (self.gNa * (self.ENa - self.V) + \
                                    self.gK * (self.EK - self.V) + \
                                    self.gl*(self.El - self.V) + \
                                    self.gKa * (self.EK - self.V) + \
                                    self.gH * (self.EH - self.V) - \
                                    self.Isyn + self.Iext)
    
            self.m = self.alpha_m() / (self.alpha_m() + self.beta_m())
            self.n = self.n_integrate(dt)
            self.h = self.h_integrate(dt)
            self.a = self.a_integrate(dt)
            self.b = self.b_integrate(dt)
            self.r = self.r_integrate(dt)
            
            self.gNa = self.gbarNa * self.m * self.m * self.m * self.h
            self.gK = self.gbarK * self.n * self.n * self.n * self.n
            self.gKa = self.gbarKa * self.a * self.b
            self.gH = self.gbarH * self.r 
            self.Iext = np.random.normal(self.Iextmean, self.Iextvarience) 
    
            self.Isyn = 0
            i += 1
            t += dt
  

########
    cpdef checkFired(self, double t_):
    
        if (self.V >= self.th and self.countSp):
            self.firing = np.append(self.firing, t_)
            self.countSp = False
        
        if (self.V < self.th):
            self.countSp = True 
            
    def addIsyn(self, double Isyn):
        self.Isyn += Isyn   
            
cdef class FS_neuron(OriginCompartment):
    cdef double Capacity, Iextmean, Iextvarience, ENa, EK, El
    cdef double gbarNa, gbarK, gl, fi
    cdef double th
    cdef bool countSp
    cdef double m, h, n
    cdef double gNa, gK
    cdef double distance
    
    def __cinit__(self, params):
         self.V = params["V0"]
         self.Iextmean = params["Iextmean"]        
         self.Iextvarience = params["Iextvarience"]
         self.ENa = params["ENa"]
         self.EK = params["EK"]
         self.El = params["El"]
         self.gbarNa = params["gbarNa"]
         self.gbarK = params["gbarK"]
         self.gl = params["gl"]   
         self.fi = params["fi"]
         
         self.Iext = np.random.normal(self.Iextmean, self.Iextvarience) 
         
         self.Vhist = np.array([])
         self.LFP = np.array([])
         self.distance = np.random.normal(200, 10)
         self.firing = np.array([])
         
         self.m = self.alpha_m() / (self.alpha_m() + self.beta_m())
         self.n = self.alpha_n() / (self.alpha_n() + self.beta_n())
         self.h = self.alpha_h() / (self.alpha_h() + self.beta_h())

         self.gNa = self.gbarNa * self.m * self.m * self.m * self.h
         self.gK = self.gbarK * self.n * self.n * self.n * self.n

         self.Isyn = 0
         self.countSp = True
         self.th = -20
         
    cdef double getV(self):
        return self.V + 60
        
    def getLFP(self):
        return 0

    cdef double alpha_m(self):
         cdef double  x = -0.1 * (self.V + 33)
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
#######np.

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
    cpdef integrate (self, double dt, double duraction):

        cdef double t = 0
        cdef double i = 0
        while (t < duraction):
            self.Vhist = np.append(self.Vhist, self.V)
          
            self.V = self.V + dt * (self.gNa * (self.ENa - self.V) + self.gK * (self.EK - self.V) + self.gl*(self.El - self.V) - self.Isyn + self.Iext)
    
            self.m = self.alpha_m() / (self.alpha_m() + self.beta_m())
            self.n = self.n_integrate(dt)
            self.h = self.h_integrate(dt)
            
            self.gNa = self.gbarNa * self.m * self.m * self.m * self.h
            self.gK = self.gbarK * self.n * self.n * self.n * self.n
    
            self.Iext = np.random.normal(self.Iextmean, self.Iextvarience) 
    
            self.Isyn = 0
            i += 1
            t += dt
  

########
    cpdef checkFired(self, double t_):
    
        if (self.V >= self.th and self.countSp):
            self.firing = np.append(self.firing, t_)
            self.countSp = False
        
        if (self.V < self.th):
            self.countSp = True
    
    def addIsyn(self, double Isyn):
        self.Isyn += Isyn

cdef class ClusterNeuron(FS_neuron):
    cdef double gbarKS, gbarH, gKS, gH, Eh, H, p, q
    def __cinit__(self, params):

        self.Eh = params["Eh"]
        self.gbarKS = params["gbarKS"]
        self.gbarH = params["gbarH"]
        self.H = 1 / (1 + exp( (self.V + 80) / 10) )
        self.p = 1 / (1 + exp( -(self.V + 34) / 6.5) )
        self.q = 1 / (1 + exp( (self.V + 65) / 6.6) )
        self.gKS = self.gbarKS * self.p * self.q
        self.gH = self.gbarH * self.H
        
    cpdef integrate(self, double dt, double duration):
        cdef double t = 0
        cdef int i = 0
        while (t < duration):
            self.Vhist = np.append(self.Vhist, self.V)
            self.V = self.V + dt * (self.gNa * (self.ENa - self.V) + 
                                      self.gK * (self.EK - self.V) + 
                                      self.gKS * (self.EK - self.V) + 
                                      self.gH * (self.Eh - self.V) + 
                                      self.gl*(self.El - self.V) - 
                                      self.Isyn + self.Iext)
            self.m = self.alpha_m() / (self.alpha_m() + self.beta_m())
            self.n = self.n_integrate(dt)
            self.h = self.h_integrate(dt)
            self.H = self.H_integrate(dt)
            self.p = self.p_integrate(dt)
            self.q = self.q_integrate(dt)
              
            self.gNa = self.gbarNa * self.m * self.m * self.m * self.h
            self.gK = self.gbarK * self.n * self.n * self.n * self.n
            self.gH = self.gbarH * self.H
            self.gKS = self.gbarKS * self.p * self.q
              
              
            self.Iext = np.random.normal(self.Iextmean, self.Iextvarience) 
            self.Isyn = 0
            i += 1
            t += dt
            
    cdef double H_integrate(self, double dt):
        cdef double H_0 = 1 / (1 + exp( (self.V + 80) / 10) )
        cdef double tau_H = ( 200 / (exp( (self.V + 70) / 20) + exp( -(self.V + 70) / 20))) + 5
        return H_0 - (H_0 - self.H) * exp(-dt/tau_H)
          
    cdef double p_integrate(self, double dt):
        cdef double p_0 = 1 / (1 + exp(-(self.V + 34) / 6.5) )
        cdef double tau_p = 6
        return p_0 - (p_0 - self.p) * exp(-dt/tau_p)
          
    cdef double q_integrate(self, double dt):
        cdef double q_0 = 1 / ( 1 + exp( (self.V + 65) / 6.6) )
        cdef double tau_q0 = 100
        cdef double tau_q = tau_q0 * (1 + ( 1 / (1 + exp( -(self.V + 50) / 6.8) ) ) )
        return q_0 - (q_0 - self.q) * exp(-dt/tau_q)
####################      
cdef class PyramideCA1Compartment(OriginCompartment):
    cdef double Capacity, Iextmean, Iextvarience, ENa, EK, El, ECa, CCa, sfica, sbetaca
    cdef double gbarNa, gbarK_DR, gbarK_AHP, gbarK_C, gl, gbarCa
    cdef double th
    cdef bool countSp
    cdef double m, h, n, s, c, q
    cdef double INa, IK_DR, IK_AHP, IK_C, ICa, Il
    cdef double distance
    
    def __cinit__(self, params):
        self.V = params["V0"]
        self.Capacity = params["C"]
        
        self.Iextmean = params["Iextmean"]        
        self.Iextvarience = params["Iextvarience"]
        
        self.ENa = params["ENa"]
        self.EK = params["EK"]
        self.El = params["El"]
        self.ECa = params["ECa"]
        
        self.CCa = params["CCa"]
        self.sfica = params["sfica"]
        self.sbetaca = params["sbetaca"]
        
        self.gbarNa = params["gbarNa"]
        self.gbarK_DR = params["gbarK_DR"]

        self.gbarK_AHP = params["gbarK_AHP"]        
        self.gbarK_C = params["gbarK_C "]        

        self.gl = params["gl"]
        self.gbarCa = params["gbarCa"]
        
        self.Vhist = np.array([])
        self.LFP = np.array([])
        self.distance = np.random.normal(8, 2)
        
        self.firing = np.array([])
        self.th = self.El + 40
        
        self.m = self.alpha_m() / (self.alpha_m() + self.beta_m())
        self.h = self.alpha_h() / (self.alpha_h() + self.beta_h())
        self.n = self.alpha_n() / (self.alpha_n() + self.beta_n())
        self.s = self.alpha_s() / (self.alpha_s() + self.beta_s())
        self.c = self.alpha_c() / (self.alpha_c() + self.beta_c())
        self.q = self.alpha_q() / (self.alpha_q() + self.beta_q())
        
        self.calculate_currents()

    cdef void calculate_currents(self):
        self.Il = self.gl * (self.V - self.El)
        self.INa = self.gbarNa * self.m * self.m * self.h * (self.V - self.ENa)
        self.IK_DR = self.gbarK_DR * self.n * (self.V - self.EK)
        self.IK_AHP = self.gbarK_AHP * self.q * (self.V - self.EK)
        self.IK_C = self.gbarK_C * self.c * (self.V - self.EK)
        
        
        cdef double tmp = self.CCa / 250.0
        if (tmp < 1):
            self.IK_C *= tmp    
        
        self.ICa = self.gbarCa * self.s * self.s * (self.V - self.ECa)
        self.Iext = np.random.normal(self.Iextmean, self.Iextvarience)
        self.Isyn = 0
        self.Icoms = 0

    cdef double alpha_m(self):
        cdef double x = 13.1 - self.V
        if (x == 0):
            x = 0.000001
        cdef double alpha = 0.32 * x / (exp(0.25 * x) - 1)
        return alpha
        
        
    cdef double beta_m(self):
        cdef double x = self.V - 40.1
        if (x == 0):
            x = 0.00001
        cdef double beta = 0.28 * x / (exp(0.2 * x) - 1)
        return beta
        
    cdef double alpha_h(self):
        cdef double alpha = 0.128 * exp((17 - self.V) / 18)
        return alpha
        
    cdef double beta_h(self):
        cdef double x = 40 - self.V 
        if (x == 0):
            x = 0.00001
        cdef double beta = 4 / (exp(0.2 * x) + 1)
        return beta

    cdef double alpha_n(self):
        cdef double x = 35.1 - self.V
        if (x == 0):
            x = 0.00001
        cdef double alpha = 0.016 * x / (exp(0.2 * x) - 1)
        return alpha

    cdef double beta_n(self):
        cdef double beta = 0.25 * exp(0.5 - 0.025 * self.V)
        return beta
        
    cdef double alpha_s(self):
        cdef double x = self.V - 65
        cdef double alpha = 1.6 / (1 + exp(-0.072 * x))
        return alpha
    
    cdef double beta_s(self):
        cdef double x = self.V - 51.1
        if (x == 0):
            x = 0.00001
        cdef double beta = 0.02 * x / (exp(0.2 * x) - 1)
        return beta

    cdef double alpha_c(self):
        cdef double alpha
        if(self.V > 50):
            alpha = 2 * exp((6.5 - self.V)/27)
        else:
            alpha = exp( ((self.V - 10)/11) - ((self.V - 6.5)/27) ) / 18.975   
        return alpha
    
    cdef double beta_c(self):
        cdef double beta
        if (self.V > 0):
            beta = 0
        else:
            beta = 2 * exp((6.5 - self.V)/27) - self.alpha_c()
        return beta
    
    cdef double alpha_q(self):
        cdef double alpha = 0.00002 * self.CCa
        if (alpha > 0.01):
            alpha = 0.01
        return alpha
    
    cdef double beta_q(self):
        return 0.001
    

    cdef double h_integrate(self, double dt):
        cdef double h_0 = self.alpha_h() / (self.alpha_h() + self.beta_h())
        cdef double tau_h = 1 / (self.alpha_h() + self.beta_h())
        return h_0 - (h_0 - self.h) * exp(-dt / tau_h)


    cdef double n_integrate(self, double dt):
        cdef double n_0 = self.alpha_n() / (self.alpha_n() + self.beta_n() )
        cdef double tau_n = 1 / (self.alpha_n() + self.beta_n())
        return n_0 - (n_0 - self.n) * exp(-dt / tau_n)
        
    cdef double s_integrate(self, double dt):
        cdef double s_0 = self.alpha_s() / (self.alpha_s() + self.beta_s() )
        cdef double tau_s = 1 / (self.alpha_s() + self.beta_s())
        return s_0 - (s_0 - self.s) * exp(-dt / tau_s)
    
    cdef double c_integrate(self, double dt):
        cdef double c_0 = self.alpha_c() / (self.alpha_c() + self.beta_c() )
        cdef double tau_c = 1 / (self.alpha_c() + self.beta_c())
        return c_0 - (c_0 - self.c) * exp(-dt / tau_c)
    
    cdef double q_integrate(self, double dt):
        cdef double q_0 = self.alpha_q() / (self.alpha_q() + self.beta_q() )
        cdef double tau_q = 1 / (self.alpha_q() + self.beta_q())
        return q_0 - (q_0 - self.q) * exp(-dt / tau_q)
    
    cdef double CCa_integrate(self, double dt):
        cdef double k1 = self.CCa
        cdef double k2 = k1 + 0.5 * dt * (- self.sfica * self.ICa - self.sbetaca * k1)
        cdef double k3 = k2 + 0.5 * dt * (- self.sfica * self.ICa - self.sbetaca * k2)
        cdef double k4 = k1 + dt * (- self.sfica * self.ICa - self.sbetaca * k1)        
        return (k1 + 2*k2 + 2*k3 + k4) / 6

    cpdef integrate(self, double dt, double duration):
        cdef double t = 0
        while (t < duration):
            self.Vhist = np.append(self.Vhist, self.V)
            
            I = -self.Il - self.INa - self.IK_DR - self.IK_AHP - self.IK_C - self.ICa - self.Isyn - self.Icoms + self.Iext
            lfp = (I + self.Icoms) / (4 * np.pi * 0.3)
            # self.Isyn
            
           
            self.LFP = np.append(self.LFP, lfp)
            
            # if (self.Isyn > 10 or self.Isyn < -10):
            #    print (self.Isyn)
            
            self.V += dt * I / self.Capacity
     
            self.m = self.alpha_m() / (self.alpha_m() + self.beta_m())
            self.h = self.h_integrate(dt)
            self.n = self.n_integrate(dt)
            self.s = self.s_integrate(dt)
            self.c = self.c_integrate(dt)
            self.q = self.q_integrate(dt)
            self.CCa = self.CCa_integrate(dt)
     
            self.calculate_currents()
            self.Isyn = 0
            t += dt
    
    
    def checkFired(self, double t):
    
        if (self.V >= self.th and self.countSp):
            self.firing = np.append(self.firing, t)
            self.countSp = False
        
        if (self.V < self.th):
            self.countSp = True        
        
cdef class IntercompartmentConnection:
    cdef OriginCompartment comp1
    cdef OriginCompartment comp2
    cdef double g, p
    def __cinit__(self, OriginCompartment comp1, OriginCompartment comp2, double g, double p):
        self.comp1 = comp1
        self.comp2 = comp2
        self.g = g
        self.p = p
    
    def activate(self):
        
        cdef double Icomp1= (self.g / self.p) * (self.comp1.getV() - self.comp2.getV())
        cdef double Icomp2 = (self.g/(1 - self.p)) * (self.comp2.getV() - self.comp1.getV())
        
        self.comp1.addIcoms(Icomp1)
        self.comp2.addIcoms(Icomp2)       

cdef class ComplexNeuron:
    cdef dict compartments # map [string, OriginCompartment*] compartments
    cdef list connections # vector [IntercompartmentConnection*] connections
    
    def __cinit__(self, list compartments, list connections):
        self.compartments = dict()
        
        for comp in compartments:
            key, value = comp.popitem()
            self.compartments[key] = PyramideCA1Compartment(value)
        

        self.connections = []
        for conn in connections:
            self.connections.append(IntercompartmentConnection(self.compartments[conn["compartment1"]], self.compartments[conn["compartment2"]], conn["g"], conn["p"]   ) )
        
    def getCompartmentsNames(self):
        return self.compartments.keys()
    
    def integrate(self, double dt, double duration):
        cdef double t = 0
        
        while(t < duration):
            for p in self.compartments.values():
                p.integrate(dt, dt)
                
            for c in self.connections:
                c.activate()
            
            t += dt
            
    def getCompartmentByName(self, name): 
        return self.compartments[name]
        
    
cdef class OriginSynapse:
    cdef OriginCompartment pre
    cdef OriginCompartment post
    cdef double W
    
    def __cinit__(self, OriginCompartment pre, OriginCompartment post, params):
        pass
    
    cpdef integrate(self, double dt):
        pass


cdef class SimpleSynapse(OriginSynapse):
    cdef double Erev, tau, S, gbarS
    
    def __cinit__(self, OriginCompartment pre, OriginCompartment post, params):
      
        self.pre = pre
        self.post = post
        
        self.Erev = params["Erev"]
        self.gbarS = params["gbarS"]
        self.tau = params["tau"]
        self.W = params["w"]
        self.S = 0


    cpdef integrate(self, double dt):
        cdef double Vpre = self.pre.getV() # V of pre neuron
        
        if (Vpre > 40):
            self.S = 1
 
        if ( self.S < 0.005 ):
            self.S = 0
            return
    
        
            
        cdef double Vpost = self.post.getV()
        cdef double Isyn = self.W * self.gbarS * self.S * (Vpost - self.Erev)
        self.post.addIsyn(Isyn) #  Isyn for post neuron
        
        
        cdef double k1 = self.S
        cdef double k2 = k1 - 0.5 * dt * (self.tau * k1)
        cdef double k3 = k2 - 0.5 * dt * (self.tau * k2)
        cdef double k4 = k1 - dt * (self.tau * k1)
        
        self.S = (k1 + 2*k2 + 2*k3 + k4) / 6.0 

cdef class ComplexSynapse(OriginSynapse):
    cdef double teta, K, alpha_s, beta_s, gbarS, S, Erev, delay
    cdef queue [double] v_delay
    
    def __cinit__(self, OriginCompartment pre, OriginCompartment post, params):
        self.pre = pre
        self.post = post
        
        self.Erev = params["Erev"]
        self.gbarS = params["gbarS"]
        self.teta = params["teta"]
        self.K = params["K"]
        self.alpha_s = params["alpha_s"]
        self.beta_s = params["beta_s"]        
        self.W = params["w"]
        self.S = 0
        self.delay = params["delay"]
        if (self.delay > 0):
            for idx in range(int(self.delay)):
                self.v_delay.push(-65)
        
    cpdef integrate(self, double dt):
            
        cdef double Vpre = self.pre.getV()
        if (self.delay > 0):
            self.v_delay.push(Vpre) # V of pre neuron
            Vpre = self.v_delay.front()
            self.v_delay.pop()
            
            
        if (Vpre < 40 and self.S < 0.005):
            self.S = 0
            return
        Vpre -= 60
        cdef double Vpost = self.post.getV()
        cdef double F = 1 / (1 + exp( -(Vpre - self.teta) / self.K ) )
        cdef double S_0 = self.alpha_s * F / (self.alpha_s * F + self.beta_s)
        cdef double tau_s = 1 / (self.alpha_s * F + self.beta_s)
        self.S = S_0 - (S_0 - self.S) * exp( -dt/tau_s )
        cdef double Isyn = self.W * self.gbarS * self.S * (Vpost - self.Erev)
        self.post.addIsyn(Isyn)

cdef class SimpleSynapseWithDelay(OriginSynapse):
    cdef double Erev, tau, S, gbarS, delay

    cdef queue [double] v_delay
    def __cinit__(self, OriginCompartment pre, OriginCompartment post, params):
      
        self.pre = pre
        self.post = post
        
        self.Erev = params["Erev"]
        self.gbarS = params["gbarS"]
        self.tau = params["tau"]
        self.W = params["w"]
        self.delay = params["delay"]
        
        for idx in range(int(self.delay)):
            self.v_delay.push(-65)
        
        self.S = 0


    cpdef integrate(self, double dt):
        # cdef int idx_delay = -int(self.delay + dt) / dt
        cdef double Vpre = self.pre.getV()
        self.v_delay.push(Vpre) # V of pre neuron
        Vpre = self.v_delay.front()
        self.v_delay.pop()
        
        if (Vpre > 40):
            self.S = 1
 
        if ( self.S < 0.005 ):
            self.S = 0
            return
    
        
            
        cdef double Vpost = self.post.getV()
        cdef double Isyn = self.W * self.gbarS * self.S * (Vpost - self.Erev)
        self.post.addIsyn(Isyn) #  Isyn for post neuron
        
        
        cdef double k1 = self.S
        cdef double k2 = k1 - 0.5 * dt * (self.tau * k1)
        cdef double k3 = k2 - 0.5 * dt * (self.tau * k2)
        cdef double k4 = k1 - dt * (self.tau * k1)
        
        self.S = (k1 + 2*k2 + 2*k3 + k4) / 6.0

cdef class Network:
    cdef list neurons
    cdef list synapses

    cdef double t
    def __cinit__(self, neuron_params, synapse_params):
        self.neurons = list()
        self.synapses = list()
        self.t = 0
        cdef int idx, length
        length = len(neuron_params)
        for idx in range(length):
            if (neuron_params[idx]["type"] == "pyramide"):
                neuron = ComplexNeuron(neuron_params[idx]["compartments"], neuron_params[idx]["connections"])
                
            if (neuron_params[idx]["type"] == "FS_Neuron"):
                neuron = FS_neuron(neuron_params[idx]["compartments"])
                
            if (neuron_params[idx]["type"] == "ClusterNeuron"):
                neuron = ClusterNeuron(neuron_params[idx]["compartments"])
                
            if (neuron_params[idx]["type"] == "olm_cell"):
                neuron = OLM_cell(neuron_params[idx]["compartments"])
            
            if (neuron_params[idx]["type"] == "CosSpikeGenerator"):
                neuron = CosSpikeGenerator(neuron_params[idx]["compartments"])
                
            if (neuron_params[idx]["type"] == "PoisonSpikeGenerator"):
                neuron = PoisonSpikeGenerator(neuron_params[idx]["compartments"])

            self.neurons.append(neuron)
        length = len(synapse_params)
        
        idx = 0
        while (idx < length):
            if (synapse_params[idx]["type"] == "SimpleSynapse"):
                synapse = SimpleSynapse(self.neurons[synapse_params[idx]["pre_ind"]].getCompartmentByName(synapse_params[idx]["pre_compartment_name"]), self.neurons[synapse_params[idx]["post_ind"]].getCompartmentByName(synapse_params[idx]["post_compartment_name"]), synapse_params[idx]["params"] )

            if (synapse_params[idx]["type"] == "SimpleSynapseWithDelay"):
                synapse = SimpleSynapseWithDelay(self.neurons[synapse_params[idx]["pre_ind"]].getCompartmentByName(synapse_params[idx]["pre_compartment_name"]), self.neurons[synapse_params[idx]["post_ind"]].getCompartmentByName(synapse_params[idx]["post_compartment_name"]), synapse_params[idx]["params"] )
            
            if (synapse_params[idx]["type"] == "ComplexSynapse"):
                synapse = ComplexSynapse(self.neurons[synapse_params[idx]["pre_ind"]].getCompartmentByName(synapse_params[idx]["pre_compartment_name"]), self.neurons[synapse_params[idx]["post_ind"]].getCompartmentByName(synapse_params[idx]["post_compartment_name"]), synapse_params[idx]["params"] )
            
            
            self.synapses.append(synapse)
            idx += 1
    
    cpdef integrate(self, double dt, double duration, iext_function):
        
        cdef double Iext_model
        cdef int NN = len(self.neurons)
        cdef int NS = len(self.synapses)
        cdef int s_ind = 0
        cdef int neuron_ind = 0
        while(self.t < duration):
            #with nogil, cython.boundscheck(False), cython.wraparound(False):
                for neuron_ind in range(NN):
                    
                    self.neurons[neuron_ind].integrate(dt, dt)
                    """
                    for compartment_name in n.getCompartmentsNames():
                        Iext_model = iext_function(neuron_ind, compartment_name, t)
                        n.getCompartmentByName(compartment_name).addIsyn( Iext_model )
                    """
                    
                    
                    self.neurons[neuron_ind].getCompartmentByName("soma").checkFired(self.t)
                    
                for s_ind in range(NS):
                    self.synapses[s_ind].integrate(dt)
                

                self.t += dt
            
    def getVhist(self):
        V = []
        for idx, n in enumerate(self.neurons):
            Vn = dict()
            for key in n.getCompartmentsNames():
                Vn[key] = n.getCompartmentByName(key).getVhist()
            V.append(Vn)
        return V

    def getLFP(self, layer_name="soma"):
        lfp = 0
        for idx, n in enumerate(self.neurons):
            # lfp += n.getCompartmentByName("soma").getLFP()
            
            
            for key in n.getCompartmentsNames():
                # Vn[key] = n.getCompartmentByName(key).getVhist()
                Vext =  n.getCompartmentByName(key).getLFP()
                if (key == "soma"):
                    lfp -= Vext
                else:
                    lfp -= Vext / 10

        return lfp

    def getfullLFP(self, layer_name="soma"):
        
        lfp = []
        for idx, n in enumerate(self.neurons):
            
            keys = n.getCompartmentsNames()
            if not("dendrite" in keys):
                continue
            
            for key in keys:
                lfp.append({})
                Vext =  -1.0 * n.getCompartmentByName(key).getLFP()
                if (key == "soma"):
                    lfp[-1]["soma"] = Vext
                else:
                    lfp[-1]["dendrite"] = Vext
        return lfp

    def getFiring(self):
        firing = np.empty((2, 0), dtype=np.float64)
        for idx, n in enumerate(self.neurons):
            fired = n.getCompartmentByName("soma").getFiring()
            fired_n = np.zeros_like(fired) + idx + 1
            firing = np.append(firing, [fired, fired_n], axis=1)
            
            
        return firing



        
    def addIextbyT(self, double t):
        pass 
        

cpdef testqueue():
    cdef queue [double] q
    q.push(2.6)
    q.push(3.6)
    
    cdef double a = q.front()
    q.pop()
    print (a)
    print (q.size())
    
    
