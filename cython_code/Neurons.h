
#include <cmath>
#include <map>
#include <string>
#include <iostream>
#include <vector>

using namespace std;

namespace complex_neurons {
    class Compartment {
    public:
        Compartment(map <string, double> params);
        ~Compartment();
        void integrate(double dt, double duration);
        void checkFired(double t_);
        double getV();
        vector<double> getVhist();
        void addIsyn(double Isyn);
        void setIext(double Iext);
        vector<double>  getFiring();
        
    protected:
		double V, Capacity, Iextmean, Iextvarience, ENa, EK, El, ECa, CCa, sfica, sbetaca;
		double gbarNa, gbarK_DR, gbarK_AHP, gbarK_C, gl, gbarCa;
		double th;
		bool countSp;
		double m, h, n, s, c, q;
		double Iext, Isyn, INa, IK_DR, IK_AHP, IK_C, ICa, Il;
		vector <double> Vhist;
		vector <double> firing;
        
        void calculate_currents();
		double alpha_m();
		double beta_m();
        double alpha_h();
        double beta_h();
        double alpha_n();
        double beta_n();
        double alpha_s();
        double beta_s();
        double alpha_c();
        double beta_c();
        double alpha_q();
        double beta_q();
        double h_integrate(double dt);
        double n_integrate(double dt);
        double s_integrate(double dt);
        double c_integrate(double dt);
        double q_integrate(double dt);
        double CCa_integrate(double dt);
   
    };
}


