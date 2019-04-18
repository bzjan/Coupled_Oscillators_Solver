
#include <cmath>
#include <vector>								// std::vector
#include <iostream>								// std::cout, std::endl, for debugging
#include <complex>								// std::complex<Real>, std::exp()

#include "../params.hpp"						// struct params
#include "./solver_cpu.hpp"						// struct host_pointers




void nonlocal_homo_phase(Real* c, Real* cnew, Real couplecoeff, params &p, Real* kernel){
	
	int idx=0, jx=0;
	Real sum=0.0;
	
	if(p.bc=="periodic"){
		switch(p.spaceDim){
			case 1:
				#pragma omp parallel for private(idx,sum,jx)
				for(int x=0; x<p.nx; x++){
					idx = x;								// linearized 1d index = position of current thread in array
					
					sum=0.0;
					for(int kx=0; kx<p.kdia; kx++){
						jx=x+kx-p.kradius;
						if(jx<0){ jx+=p.nx; } else if(jx>=p.nx){ jx-=p.nx; }
						sum += kernel[kx]*sin(c[jx] - c[idx] - couplecoeff);
					}
					
					// calculation
					cnew[idx] += sum;
				}
				break;
				
				
			default: printf("spaceDim is not chosen correctly for diffusion! Program Abort!"); exit(EXIT_FAILURE); break;
		}
	}else if(p.bc=="neumann"){
		switch(p.spaceDim){
			case 1:
				#pragma omp parallel for private(idx,jx,sum)
				for(int x=0; x<p.nx; x++){
					idx = x;								// linearized 1d index = position of current thread in array
					
					sum=0.0;
					for(int kx=0; kx<p.kdia; kx++){
						jx=x+kx-p.kradius;
						if(jx >= 0 && jx < p.nx){
							sum += kernel[kx]*sin(c[jx] - c[idx] - couplecoeff);
						}
					}
					
					// calculation
					cnew[idx] += sum;
				}
				break;
				
				
			default:
				printf("spaceDim is not chosen correctly for diffusion! Program Abort!"); exit(EXIT_FAILURE);
				break;
		}
	}
}




void nonlocal_homo_zbke2k(Real* v, Real* cnew, Real* couplecoeff, params &p, Real* kernel){
	
	int idx=0, jx=0;
	double sum=0.0;
	
	if(p.bc=="periodic"){
		switch(p.spaceDim){
			case 1:
				#pragma omp parallel for private(idx,sum,jx)
				for(int x=0; x<p.nx; x++){
					idx = x;								// linearized 1d index = position of current thread in array
					
					sum=0.0;
					for(int kx=0; kx<p.kdia; kx++){
						jx=x+kx-p.kradius;
						if(jx<0){ jx+=p.nx; } else if(jx>=p.nx){ jx-=p.nx; }
						sum += kernel[kx]*v[jx];
					}
					sum -= p.ksum*v[idx];
					
					// calculation
					cnew[idx] += couplecoeff[0]*sum;
					cnew[p.n+idx] += couplecoeff[1]*sum;
				}
				break;
				
				
			default: printf("spaceDim is not chosen correctly for diffusion! Program Abort!"); exit(EXIT_FAILURE); break;
		}
	}else if(p.bc=="neumann"){
		switch(p.spaceDim){
			case 1:
				#pragma omp parallel for private(idx,jx,sum)
				for(int x=0; x<p.nx; x++){
					idx = x;								// linearized 1d index = position of current thread in array
					
					sum=0.0;
					for(int kx=0; kx<p.kdia; kx++){
						jx=x+kx-p.kradius;
						if(jx >= 0 && jx < p.nx){
							sum += kernel[kx]*(v[jx]-v[idx]);
						}
					}
					
					// calculation
					cnew[idx] += couplecoeff[0]*sum;
					cnew[p.n+idx] += couplecoeff[1]*sum;
				}
				break;
				
				
			default: printf("spaceDim is not chosen correctly for diffusion! Program Abort!"); exit(EXIT_FAILURE); break;
		}
	}
}



// cdelay = v-component only
// c = v-component only
void nonlocal_delay_homo_zbke2k(Real* c, Real* cnew, Real* cdelay, Real* couplecoeff, params &p, Real* kernel){
	
	
	int idx=0, jx=0;
	double sum=0.0;
	
	if(p.bc=="periodic"){
		switch(p.spaceDim){
			case 1:
				#pragma omp parallel for private(idx,sum,jx)
				for(int x=0; x<p.nx; x++){
					idx = x;											// linearized 1d index = position of current thread in array
					
					sum=0.0;
					for(int kx=0; kx<p.kdia; kx++){
						jx=x+kx-p.kradius;
						if(jx<0){ jx+=p.nx; } else if(jx>=p.nx){ jx-=p.nx; }
						sum += kernel[kx]*cdelay[jx];
					}
					sum -= p.ksum*c[idx];
					
					// calculation
					cnew[idx] += couplecoeff[0]*sum;
					cnew[p.n+idx] += couplecoeff[1]*sum;
				}
				break;
				
			default: 
				printf("spaceDim is not chosen correctly for nonlocal_delay_homo_zbke2k! Program Abort!"); 
				exit(EXIT_FAILURE); 
				break;
		}
	}else if(p.bc=="neumann"){
		switch(p.spaceDim){
			case 1:
				#pragma omp parallel for private(idx,jx,sum)
				for(int x=0; x<p.nx; x++){
					idx = x;											// linearized 1d index = position of current thread in array
					
					sum=0.0;
					for(int kx=0; kx<p.kdia; kx++){
						jx=x+kx-p.kradius;
						if(jx >= 0 && jx < p.nx){
							sum += kernel[kx]*(cdelay[jx]-c[idx]);
						}
					}
					
					// calculation
					cnew[idx] += couplecoeff[0]*sum;
					cnew[p.n+idx] += couplecoeff[1]*sum;
				}
				break;
				
			default:
				printf("spaceDim is not chosen correctly for nonlocal_delay_homo_zbke2k! Program Abort!");
				exit(EXIT_FAILURE);
				break;
		}
	}
}





// global coupling via order parameter
void network_global_coupling_phase(Real* c, Real* cnew, params &p){
	
	// 1) get complex order parameter
	const std::complex<Real> ii(0, 1);		// imaginary unit
	std::complex<Real> R(0, 0);				// complex order parameter
	#pragma omp parallel for
	for(int i=0; i<p.n; i++) R += std::exp(ii * c[i]);
	R /= p.n;
	
	// 2) coupling update: cnew_i += dt*K*r*sin(psi - phi_i)
	Real r = abs(R);						// magnitude of order parameter
	Real psi = arg(R);						// phase of order parameter
	Real multiplier = p.dt*p.K*r;
	#pragma omp parallel for
	for(int i=0; i<p.n; i++) cnew[i] += multiplier*sin(psi-c[i]);
}





// global coupling via order parameter
void network_global_coupling_frustration_phase(Real* c, Real* cnew, params &p, modelparams &m){
	
	// 1) get complex order parameter
	const std::complex<Real> ii(0, 1);		// imaginary unit
	std::complex<Real> R(0, 0);				// complex order parameter
	#pragma omp parallel for
	for(int i=0; i<p.n; i++) R += std::exp(ii * c[i]);
	R /= p.n;
	
	// 2) coupling update: cnew_i += dt*K*r*sin(psi - phi_i - alpha)
	Real r = abs(R);						// magnitude of order parameter
	Real psi = arg(R);						// phase of order parameter
	Real multiplier = p.dt*p.K;
	#pragma omp parallel for
	for(int i=0; i<p.n; i++) cnew[i] += multiplier*r*sin(psi-c[i]-m.coupling_coeffs[0]);
}






// global coupling by summing up all v[j] and subtracting p.n*v[i]
void network_global_coupling_delay_zbke(Real* c, Real* cnew, Real* cdelay, params &p, modelparams &m){
	
	// 1) sum up all v[i]
	Real vsum = 0.0;
	Real *vdelay = cdelay + p.n;
	#pragma omp parallel for
	for(int i=0; i<p.n; i++) vsum += vdelay[i];
	
	// 2) coupling update: cnew_i += dt*K*(ku,kv)*( sum_j(v_j(t-\tau)) - n*v_i(t))
	Real *v = c + p.n;
	std::vector<Real> multipliers{p.dt*p.K*m.coupling_coeffs[0], p.dt*p.K*m.coupling_coeffs[1]};
	#pragma omp parallel for collapse(2)
	for(int j=0; j<p.ncomponents; j++){
	for(int i=0; i<p.n; i++){
		cnew[i+j*p.n] += multipliers[j]*(vsum - p.n*v[i]);
	}}
}



// global coupling by summing up all v[j] and subtracting p.n*v[i]
void network_global_coupling_zbke(Real* c, Real* cnew, params &p, modelparams &m){
	
	// 1) sum up all v[i]
	Real vsum = 0.0;
	Real *v = c + p.n;
	#pragma omp parallel for
	for(int i=0; i<p.n; i++) vsum += v[i];
	
	// 2) coupling update: cnew_i += dt*K*(ku,kv)*( sum_j(v_j) - n*v_i)
	std::vector<Real> multipliers{p.dt*p.K*m.coupling_coeffs[0], p.dt*p.K*m.coupling_coeffs[1]};
	#pragma omp parallel for collapse(2)
	for(int j=0; j<p.ncomponents; j++){
	for(int i=0; i<p.n; i++){
		cnew[i+j*p.n] += multipliers[j]*(vsum - p.n*v[i]);
	}}
}






/*
# 5-nonlocal
# 6-nonlocal,delay, 
# 1002 - global network
# 1003 - global network with frustration
# 1004 - global network with delay
*/
// coupling coefficients are rescaled properly with dt and dxx in create_kernel_and_rescale
void coupling(Real* c, Real* cnew, host_pointers *h, params &p, modelparams &m, size_t step){
	
	
	// iterate over all components by changing the offset
	// move offset in array to address different components
	if(step>=p.stepsCouplingStart){
		switch(p.couplingChoice){
			
			case 5:														// nonlocal kernel, convolution
				if(p.reactionModel==24 or p.reactionModel==2401){
					nonlocal_homo_zbke2k(c+p.n,cnew,m.coupling_coeffs,p,h->mask);
				}else if(p.reactionModel==16 or p.reactionModel==1601){
					nonlocal_homo_phase(c,cnew,m.coupling_coeffs[0],p,h->mask);
				}
				break;
			
			
			case 6:														// nonlocal, delay; fill history until delaySteps+1: only local dynamics, no coupling
				if(step<p.delayStartSteps+1){							// before delay history is full
					if(p.reactionModel==24 or p.reactionModel==2401){
						nonlocal_homo_zbke2k(c+p.n,cnew,m.coupling_coeffs,p,h->mask);
					}else if(p.reactionModel==16 or p.reactionModel==1601){
						nonlocal_homo_phase(c,cnew,m.coupling_coeffs[0],p,h->mask);
					}
				}else if(step>=p.delayStartSteps+1){					// after delay history is full
					if(p.reactionModel==24 or p.reactionModel==2401){
						nonlocal_delay_homo_zbke2k(c+p.n,cnew,h->cdelay+p.n,m.coupling_coeffs,p,h->mask);
					}else if(p.reactionModel==16 or p.reactionModel==1601){
						printf("Error: Only implemented on GPU\n"); exit(EXIT_FAILURE);
					}
				}
				break;
			
			
			case 1002:													// global network
				if(p.reactionModel==24 or p.reactionModel==2401){
					network_global_coupling_zbke(c,cnew,p,m);
				}else if(p.reactionModel==16 or p.reactionModel==1601){
					network_global_coupling_phase(c,cnew,p);
				}
				break;
			
			
			case 1003:													// global network with frustration
				if(p.reactionModel==16 or p.reactionModel==1601){
					network_global_coupling_frustration_phase(c,cnew,p,m);
				}else{
					printf("Error: coupling not implemented! (coupling @ line: %d)\n",__LINE__); exit(EXIT_FAILURE);
				}
				break;
			
			
			case 1004:													// global network with delay; fill history until delaySteps+1: only local dynamics, no coupling
				if(step<p.delayStartSteps+1){							// before delay history is full
					
					if(p.reactionModel==24 or p.reactionModel==2401){
						network_global_coupling_zbke(c,cnew,p,m);
					}else if(p.reactionModel==16 or p.reactionModel==1601){
						printf("Error: coupling not implemented! (coupling @ line: %d)\n",__LINE__); exit(EXIT_FAILURE);
					}
				}else if(step>=p.delayStartSteps+1){				// after delay history is full
					
					if(p.reactionModel==24 or p.reactionModel==2401){
						network_global_coupling_delay_zbke(c,cnew,h->cdelay,p,m);
					}else if(p.reactionModel==16 or p.reactionModel==1601){
						printf("Error: coupling not implemented! (coupling @ line: %d)\n",__LINE__); exit(EXIT_FAILURE);
					}
				
				}
				break;
			
			
			default: printf("Error: couplingChoice \"%d\" not implemented!\n",p.couplingChoice); exit(EXIT_FAILURE); break;
		}
	}
}
