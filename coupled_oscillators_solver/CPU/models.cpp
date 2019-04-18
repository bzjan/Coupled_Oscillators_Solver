
#include <cmath>
#include <vector>								// std::vector
#include <string>								// std::string


#include "../params.hpp"					// struct params
#include "./solver_cpu.hpp"						// struct host_pointers





// model 16: simple phase oscillator model; unwrapped; wrapping in analysis after simulation
void model_phase_oscillator(Real *c, Real *cnew, params &p){
	Real w = 6.28318;							// angular frequency; period = 2pi/w = 1.0
	
	#pragma omp parallel for
	for(int i=0; i<p.n; i++) cnew[i] = c[i] + p.dt*( w );
}

// model 1601: simple phase oscillator model; unwrapped; wrapping in analysis after simulation
void model_phase_oscillator_whet(Real *c, Real *cnew, Real *het, params &p){
	//~ Real w = 6.28318;							// angular frequency; period = 2pi/w = 1.0
	
	#pragma omp parallel for
	for(int i=0; i<p.n; i++) cnew[i] = c[i] + p.dt*( het[i] );
}

// model 24: ZBKE 
void model_zbke2k(Real *c, Real *cnew, params &p){
	Real ooeps1=9.090909090909091;	// 1.0f/0.11f
	//~ Real eps2=1.7e-5;
	//~ Real gamma=1.2;
	Real gammaEps2=2.04e-5;
	//~ Real eps3=1.6e-3;
	Real eps31=1.0016;
	Real alpha=0.1;
	Real beta=1.7e-5;
	Real mu=2.4e-4;
	Real q=0.7;
	Real phi=5.25e-4;
	
	Real uss=0.0;
	Real temp=0.0;

	int i2;
	#pragma omp parallel for private(i2,uss,temp)
	for(int i=0; i<p.n; i++){
		i2=i+p.n;
		uss=1.0/(4.0*gammaEps2)*(-(1.0-c[i2])+sqrt(1.0-2.0*c[i2] + c[i2]*c[i2] + 16.0*gammaEps2*c[i]));
		temp=alpha*c[i2]/(eps31-c[i2]);
		
		cnew[i]=c[i]+p.dt*( ooeps1*(phi-c[i]*c[i]-c[i]+gammaEps2*uss*uss+uss*(1.0-c[i2])+(mu-c[i])/(mu+c[i])*(q*temp+beta)) ); 			// u
		cnew[i2]=c[i2]+p.dt*( 2.0*phi + uss*(1.0-c[i2]) - temp );																		// v
	}
}


// model 2401: ZBKE with heterogeneity in q
void model_zbke2k_qhet(Real *c, Real *cnew, Real *het, params &p){
	Real ooeps1=9.090909090909091;	// 1.0f/0.11f
	//~ Real eps2=1.7e-5;
	//~ Real gamma=1.2;
	Real gammaEps2=2.04e-5;
	//~ Real eps3=1.6e-3;
	Real eps31=1.0016;
	Real alpha=0.1;
	Real beta=1.7e-5;
	Real mu=2.4e-4;
	//~ Real q=0.7;
	Real phi=5.25e-4;
	
	Real uss=0.0;
	Real temp=0.0;

	int i2;
	#pragma omp parallel for private(i2,uss,temp)
	for(int i=0; i<p.n; i++){
		i2=i+p.n;
		uss=1.0/(4.0*gammaEps2)*(-(1.0-c[i2])+sqrt(1.0-2.0*c[i2] + c[i2]*c[i2] + 16.0*gammaEps2*c[i]));
		temp=alpha*c[i2]/(eps31-c[i2]);
		
		cnew[i]=c[i]+p.dt*( ooeps1*(phi-c[i]*c[i]-c[i]+gammaEps2*uss*uss+uss*(1.0-c[i2])+(mu-c[i])/(mu+c[i])*(het[i]*temp+beta)) ); 			// u, het
		cnew[i2]=c[i2]+p.dt*( 2.0*phi + uss*(1.0-c[i2]) - temp );																				// v
	}
}



void reaction(Real* c, Real* cnew, host_pointers *h, params &p){
	
	switch(p.reactionModel){
		case 16:	model_phase_oscillator(c,cnew,p); break;
		case 1601:	model_phase_oscillator_whet(c,cnew,h->het,p); break;
		
		case 24:	model_zbke2k(c,cnew,p); break;
		case 2401:	model_zbke2k_qhet(c,cnew,h->het,p); break;
		
		default: printf("chosen reactionModel (%d) is not implemented! Program Abort!",p.reactionModel); exit(EXIT_FAILURE); break;
	}
}
