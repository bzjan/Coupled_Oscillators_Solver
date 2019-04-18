

#include <string>								// std::string (in struct of params.hpp)
#include <vector>								// std::vector (for params.hpp)
#include <cublas_v2.h>							// cublasHandle_t (in solver_gpu.hu)


#include "../params.hpp"					// struct params
#include "./solver_gpu.hu"						// struct device_pointers



// simple phase oscillator model; unwrapped; wrapping in analysis after simulation
__global__ void model_phase_oscillator(Real1 *c, Real1 *cnew, int len, Real dt){
	Real w=6.28318;							// angular frequency; period = 2pi/w = 1.0
	
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if(i<len){
		cnew[i].x = c[i].x + dt*( w );
	}
}


// simple phase oscillator model; unwrapped; wrapping in analysis after simulation; with heterogeneity
__global__ void model_phase_oscillator_whet(Real1 *c, Real1 *cnew, int len, Real dt, Real *het){
	//~ Real w=6.28318;							// angular frequency; period = 2pi/w = 1.0
	
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if(i<len){
		cnew[i].x = c[i].x + dt*( het[i] );
	}
}


// zbke2k - modified BZ model
// must be double, not float
__global__ void model_zbke2k(Real2 *c, Real2 *cnew, int len, Real dt){
	Real ooeps1=9.090909090909091;			// 1.0/0.11
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
	
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if(i<len){
		uss=1.0/(4.0*gammaEps2) * (-(1.0-c[i].y) + sqrt(1.0 + fma(c[i].y,c[i].y,-2.0*c[i].y) + 16.0*gammaEps2*c[i].x));
		temp=alpha*c[i].y/(eps31-c[i].y);
		
		cnew[i].x=c[i].x+dt*( ooeps1*(phi-c[i].x*c[i].x-c[i].x+gammaEps2*uss*uss+uss*(1.0-c[i].y)+(mu-c[i].x)/(mu+c[i].x)*(q*temp+beta)) );			// u
		cnew[i].y=c[i].y+dt*(2.0*phi + uss*(1.0-c[i].y) - temp);																					// v
	}
}


// zbke2k - modified BZ model, heterogenous q
// must be double, not float
__global__ void model_zbke2k_qhet(Real2 *c, Real2 *cnew, int len, Real dt, Real *het){
	Real ooeps1=9.090909090909091;			// 1.0/0.11
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
	
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if(i<len){
		uss=1.0/(4.0*gammaEps2) * (-(1.0-c[i].y) + sqrt(1.0 + fma(c[i].y,c[i].y,-2.0*c[i].y) + 16.0*gammaEps2*c[i].x));
		temp=alpha*c[i].y/(eps31-c[i].y);
		
		cnew[i].x=c[i].x+dt*( ooeps1*(phi-c[i].x*c[i].x-c[i].x+gammaEps2*uss*uss+uss*(1.0-c[i].y)+(mu-c[i].x)/(mu+c[i].x)*(het[i]*temp+beta)) );			// u, het
		cnew[i].y=c[i].y+dt*(2.0*phi + uss*(1.0-c[i].y) - temp);																							// v
	}
}



void reaction(device_pointers *d, params &p, streams *s){
	
	int warpsize=32;
	dim3 nblocks((p.n-1)/warpsize+1);
	dim3 nthreads(warpsize);
	
	switch(p.reactionModel){
		case 16:	model_phase_oscillator<<<nblocks,nthreads,0,s->stream1>>>((Real1 *)d->c,(Real1 *)d->cnew,p.n,p.dt); break;
		case 1601:	model_phase_oscillator_whet<<<nblocks,nthreads,0,s->stream1>>>((Real1 *)d->c,(Real1 *)d->cnew,p.n,p.dt,d->het); break;
		
		case 24:	model_zbke2k<<<nblocks,nthreads,0,s->stream1>>>((Real2 *)d->c,(Real2 *)d->cnew,p.n,p.dt); break;
		case 2401:	model_zbke2k_qhet<<<nblocks,nthreads,0,s->stream1>>>((Real2 *)d->c,(Real2 *)d->cnew,p.n,p.dt,d->het); break;
		
		default: printf("chosen reactionModel (%d) is not implemented! Program Abort!",p.reactionModel); exit(EXIT_FAILURE); break;
	}
	
	checkCUDAError("reaction()",__LINE__);
}

