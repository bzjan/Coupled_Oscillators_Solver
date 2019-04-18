/*
 * Copyright 2019 Jan Totz <jantotz@itp.tu-berlin.de>
 */

// C++ std functions
#include <vector>								// std::vector (for tip finder and params.hpp)
#include <iostream>								// std::cout, std::endl
#include <list>									// std::list (for fila finder)
#include <string>								// std::string (in struct of params.hpp)
#include <stdio.h>								// printf
#include <fstream>								// std::ofstream, std::ifstream

// CUDA libraries
#include <cublas_v2.h>							// CUDA matrix multiplication


// custom libraries
#include "../params.hpp"
#include "../Utilities/safe.hpp"
#include "./solver_gpu.hu"
#include "./vector_types_operator_overloads.hu"
#include "./models.hu"
#include "./coupling.hu"





// temporary pointers, could be expanded
struct host_pointers {

	Real *c;							// c array from host
	Real *het;							// heterogeneity array from host
	Real *mask;							// initialize nonlocal convolution kernel/mask

};




// cuda error checking function
// usage: checkCUDAError("test",__LINE__);
void checkCUDAError(const char *msg, int line){
#ifdef DEBUG
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if ( err != cudaSuccess){
		fprintf(stderr, "Cuda error: line %d: %s: %s.\n", line, msg, cudaGetErrorString( err) );
		exit(EXIT_FAILURE);
	}
#endif
}




// call examples: 
// printf("step: %zu\n",step); device_printf<<<1,1>>>((Real2*)d->c,p.n); cudaDeviceSynchronize();
// printf("c step: %zu\n",step); device_printf<<<1,1>>>((Real2*)d->c,3); cudaDeviceSynchronize();
__global__ void device_printf(Real *array, int n){
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if(idx==0){
		for(int i=0; i<n; i++) printf("%.1f ", array[i]);
		printf("\n");
	}
	__syncthreads();
}


__global__ void device_printf(Real2 *array, int n){
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if(idx==0){
		for(int i=0; i<n; i++) printf("%.1f ", array[i].x);
		printf("\n");
		for(int i=0; i<n; i++) printf("%.1f ", array[i].y);
		printf("\n");
	}
	__syncthreads();
}




// CPU pointer swap function for GPU
template <typename T>
void swapGPU(T &a, T &b){
	T temp = a;
	a = b;
	b = temp;
}






template <typename T>
__global__ void copyArrays(T *in, T *out, int len){

	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if(i<len) out[i]=in[i];

}




Real kernelfunction(int i, int i0, params &p){

	Real value=0.0;

	// exponential decay
	switch(p.spaceDim){
		case 1:
			value=p.dx*exp(-abs(i-i0)*p.dx/p.kappa);
			break;
	}

	return value;
}

void create_kernel_and_rescale(int &maskSize, host_pointers *h, params &p, modelparams &m){
	
	
	switch(p.couplingChoice){
		
		// nonlocal BZ chimera coupling: Exp(-d/kappa), d = euclidean distance
		case 5:
		case 6:
			{
				int i=0, i0=0;
				p.kdia=2*p.cutoffRange+1;
				p.kradius = p.cutoffRange;
				p.ksum = 0.0;
				switch(p.spaceDim){
					case 1:
						maskSize=p.kdia;
						i0=p.cutoffRange;
						h->mask = new Real[p.kdia];
						for(i=0; i<p.kdia; i++) h->mask[i] = kernelfunction(i,i0,p);
						for(i=0; i<maskSize; i++) h->mask[i] *= p.K*p.dt;
						break;
					
					
					default: printf("Error: spaceDim=%d not implemented.\n",p.spaceDim); exit(EXIT_FAILURE); break;
				}
				// save kernel in binary data format for later
				std::ofstream dataout;
				dataout.open(p.pthout+"/coupling_kernel.bin", std::ios::binary);
				for(int i=0; i<maskSize; i++) dataout.write((char*) &(h->mask[i]), sizeof(Real));
				dataout.close();
				for(int i=0; i<maskSize; i++) p.ksum += h->mask[i];
			}
			break;
		
		
		
		// no rescaling
		case 1002:					// global network
		case 1003:					// global network phase frustration
		case 1004:					// zbke global network with delay
			break;

		default:
			printf("Error (create_kernel_and_rescale): couplingChoice (%d) not implemented!\n",p.couplingChoice); exit(EXIT_FAILURE);
			break;
	}
}


// returns memory size of array: p.n*sizeof(data) != p.n*p.ncomponents
void getArrayMemorySize(params &p, int &array_mem, int &array_size){

	switch(p.ncomponents){
		case 1:
			array_mem=p.n*sizeof(Real1); 
			array_size=p.n; 
			break;
		case 2:
			array_mem=p.n*sizeof(Real2); 
			array_size=p.n*2; 
			break;
		default:
			printf("getArrayMemorySize: Number of components (%d) is not supported {1;2}!",p.ncomponents);
			exit(EXIT_FAILURE);
			break;
	}
}






void cleanup_GPU(Real *c, device_pointers *d, params &p){

	printf("cleanup_GPU\n");
	cudaError_t err;

	err = cudaFree(d->het);
	err = cudaFree(d->output);
	err = cudaFree(d->mask);
	if(p.delayFlag==0){ err = cudaFree(d->c); err = cudaFree(d->cnew); }
	else if(p.delayFlag==1){ err = cudaFree(d->c0); }


	// DEBUG
	if(err != cudaSuccess){
		printf("Cuda error: %s\n",cudaGetErrorString(err) );
		exit(EXIT_FAILURE);
	}
}




// manage copy operation on GPU
void copy_GPU_to_GPU(device_pointers *d, params &p, streams *s){


	cudaDeviceSynchronize();


	int warpsize=32;
	dim3 nblocks2((p.n-1)/warpsize+1,1,1);
	dim3 nthreads2(warpsize,1,1);

	switch(p.ncomponents){
		case 1: copyArrays<<<nblocks2,nthreads2,0,s->stream1>>>((Real1*)d->c,(Real1*)d->output,p.n); break;
		case 2: copyArrays<<<nblocks2,nthreads2,0,s->stream1>>>((Real2*)d->c,(Real2*)d->output,p.n); break;
		default: printf("Error in copy_GPU_to_GPU(): number of components not supported!\n"); exit(EXIT_FAILURE); break;
	}

	checkCUDAError("copyArrays invocation",__LINE__);

	cudaDeviceSynchronize();
}


void copy_GPU_to_CPU(device_pointers *d, Real *c, params &p, streams *s){

	switch(p.ncomponents){
		case 1: cudaMemcpy(c,d->output,p.n*sizeof(Real1),cudaMemcpyDeviceToHost); break;
		case 2: cudaMemcpy(c,d->output,p.n*sizeof(Real2),cudaMemcpyDeviceToHost); break;
		default: printf("Error in copy_GPU_to_CPU(): number of components not supported!\n"); exit(EXIT_FAILURE); break;
	}

}



void init_GPU(streams *s, params &p, host_pointers *h, device_pointers *d, modelparams &m){

	int array_size=0;
	int array_mem=0;
	Real needed_mem=0.0;		// memory in bytes

	// algorithm dependent improvements, did not improve speed much...
	#ifdef DOUBLE
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);					// shared memory in 64 bit mode, better for double datatypes
	#endif
	if(p.use_tiles) cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);				// get more shared memory, better for tiling

	// DEBUG: available memory
	size_t freeMem, totalMem;
	cudaMemGetInfo(&freeMem,&totalMem);
	printf("available/total GPU memory: %.2f/%.2f\n" ,freeMem/(1024.*1024.),totalMem/(1024.*1024.));

	// allocate GPU memory & move data from host to device
	getArrayMemorySize(p,array_mem,array_size);
	if(p.delayFlag==0){												// no delay
		printf("c & cnew size: %.3f MB = states: 2\n",2*array_mem/(1024.*1024.));
		needed_mem+=2*array_mem;
		if(needed_mem/(1024.*1024.)>freeMem/(1024.*1024.)){ printf("Error: Too much GPU memory required! Abort now\n"); exit(1); }
		cudaMalloc(&(d->c),array_mem);
		cudaMalloc(&(d->cnew),array_mem);
		cudaMemcpy(d->c,h->c,array_mem,cudaMemcpyHostToDevice);
	}else if(p.delayFlag==1){										// with delay
		size_t history_array_mem=array_mem*(p.delayStepsMax+1);
		printf("c0 size: %.3f MB = states: %zu\n",history_array_mem/(1024.*1024.), p.delayStepsMax+1);
		needed_mem+=history_array_mem;
		if(needed_mem/(1024.*1024.)>freeMem/(1024.*1024.)){ printf("Error: Too much GPU memory required! Abort now\n"); exit(1); }
		cudaMalloc(&(d->c0),history_array_mem);
		// init arrays for first step
		d->c = d->c0;
		cudaMemcpy(d->c,h->c,array_mem,cudaMemcpyHostToDevice);
		d->cnew = d->c0 + array_size;
	}else{
		printf("Value of delayFlag not supported!\n"); exit(EXIT_FAILURE);
	}

	printf("output size: %.3f MB\n",array_mem/(1024.*1024.));
	needed_mem+=array_mem;
	if(needed_mem/(1024.*1024.)>freeMem/(1024.*1024.)){ printf("Error: Too much GPU memory required! Abort now\n"); exit(1); }
	cudaMalloc(&(d->output),array_mem);


	// mask for convolutions and rescaling
	// add new nonlocal coupling schemes in here!
	int maskSize=0;
	create_kernel_and_rescale(maskSize,h,p,m);
	if( p.couplingChoice==5 || p.couplingChoice==6 ){
		printf("kernel size: %.3f MB\n",maskSize*sizeof(Real)/(1024.*1024.));
		needed_mem+=maskSize*sizeof(Real);
		if(needed_mem/(1024.*1024.)>freeMem/(1024.*1024.)){ printf("Error: Too much GPU memory required! Abort now\n"); exit(1); }
		cudaMalloc(&(d->mask),maskSize*sizeof(Real));
		cudaMemcpy(d->mask,h->mask,maskSize*sizeof(Real),cudaMemcpyHostToDevice);
	}

	// set coupling coefficients' values
	switch(p.ncomponents){
		case 1:
			d->coupling_coeffs1.x=m.coupling_coeffs[0];
			break;

		case 2:
			d->coupling_coeffs2.x=m.coupling_coeffs[0];
			d->coupling_coeffs2.y=m.coupling_coeffs[1];
			break;

		default: printf("Error: Number of components is not supported!"); exit(EXIT_FAILURE); break;
	}



	// use heterogeneity (het1)
	if(p.hehoflag!=0){
		int hetArraySize=p.n;
		printf("het-array size 1: %.3f MB\n",hetArraySize*sizeof(Real)/(1024.*1024.));
		needed_mem+=hetArraySize*sizeof(Real);
		if(needed_mem/(1024.*1024.)>freeMem/(1024.*1024.)){ printf("Error: Too much GPU memory required! Abort now\n"); exit(1); }
		cudaMalloc(&(d->het),hetArraySize*sizeof(Real));
		cudaMemcpy(d->het,h->het,hetArraySize*sizeof(Real),cudaMemcpyHostToDevice);
	}

	// info
	cudaMemGetInfo(&freeMem,&totalMem);
	printf("available/total GPU memory: %.2f/%.2f\n" ,freeMem/(1024.*1024.),totalMem/(1024.*1024.));
	printf("total amount of GPU memory required: %.3f MB\n",needed_mem/(1024.*1024.));
	if(!(p.spaceDim == 1000)){		// not a network or phase diagram mode
		printf("total number of threads per block: %.0f <= 1024?\n",pow(p.blockWidth,p.spaceDim));
		if(pow(p.blockWidth,p.spaceDim)>1024){ printf("Error: Using too many threads per block!\n"); exit(EXIT_FAILURE);}
	}

	checkCUDAError("init_GPU()",__LINE__);

}




void time_evolution(device_pointers *d, params &p, streams *s, size_t step, modelparams &m){
	
	switch(p.timeEvolutionChoice){
		case 0:										// explicit euler
			reaction(d,p,s);						// reaction part
			coupling(d,p,s,step,m);					// spatial coupling
			break;
		
		default: printf("Error: timeEvolutionChoice not implemented!"); exit(EXIT_FAILURE); break;
	}
}


void dynamics(device_pointers *d, params &p, streams *s, size_t step, modelparams &m){

	switch(p.delayFlag){
		case 0:											// no delay
			time_evolution(d,p,s,step,m);
			swapGPU(d->c,d->cnew);						// issue device pointer swap from host
			break;
		
		case 1:											// with delay
			time_evolution(d,p,s,step,m);
			
			
			// update pointer positions
			d->c = d->c0 + ((step+1) % (p.delayStepsMax+1))*p.n*p.ncomponents;
			d->cnew = d->c0 + ((step+2) % (p.delayStepsMax+1))*p.n*p.ncomponents;
			
			if(step>=p.delayStartSteps){
				size_t delaySteps=0;
				if(p.delayTimeIntervalUntilMaxReached==0){
					delaySteps=p.delayStepsMax;
				}else{
					delaySteps=(size_t)max((double)p.delayStepsMax,p.delayStepsMin + p.dt*(p.delayTimeMax-p.delayTimeMin)/p.delayTimeIntervalUntilMaxReached*(step-p.delayStartSteps));
				}
				d->cdelay=d->c0+((step+3) % (delaySteps+1))*p.n*p.ncomponents;
			}
			break;
		
		default: printf("Unknown value for delayFlag (rd_dynamics).\n"); break;
	}
}



void solverGPU_network(Real *c, Real *het, params &p, modelparams &m){
	
	printf("solverGPU_network\n");
	
	// init
	device_pointers d;
	host_pointers h;
	h.c = c;
	h.het = het;
	streams s;
	int untranslatedFlag=1;
	Real *ctemp = (Real *) calloc(p.n*p.ncomponents,sizeof(Real));
	
	// init classes for analysis and saving
	Safe safe(p);
	
	// save initial condition, ic=0
	translateArrayOrder(c,ctemp,p,untranslatedFlag,0);
	safe.save(ctemp,0);
	
	// prepare GPU
	init_GPU(&s,p,&h,&d,m);
	
	// time loop
	for(size_t step=0; step<=p.stepsEnd; step++){
		//~ printf("step=%d\n",step);
		dynamics(&d,p,&s,step,m);
		
		// output: save state
		if( !(step%p.stepsSaveState) and step>p.stepsSaveStateOffset ){
			
			// copy
			copy_GPU_to_GPU(&d,p,&s);
			copy_GPU_to_CPU(&d,c,p,&s);
			if(c[0]!=c[0]){ printf("step: %zu, c[0]=%f. Abort!\n",step,c[0]); exit(EXIT_FAILURE); }
			
			// translate array from concentration major to space major
			translateArrayOrder(c,ctemp,p,untranslatedFlag,0);
			
			// save
			if(!(step%p.stepsSaveState) and step>p.stepsSaveStateOffset){
				if(untranslatedFlag){ safe.save(c,step); }
				else{ safe.save(ctemp,step); }
			}
			
		}
		
		checkCUDAError("loop iteration",__LINE__);						// DEBUG
	}
	
	
	// clean up data
	cleanup_GPU(c,&d,p);
	free(ctemp);
}


void solverGPU_1d(Real *c, Real *het, params &p, modelparams &m){

	printf("solverGPU_1d\n");

	// init
	//~ cudaError_t err=cudaSuccess;
	device_pointers d;
	host_pointers h;
	h.c = c; h.het = het; 
	streams s;
	int untranslatedFlag=1;
	Real *ctemp = (Real *) calloc(p.n*p.ncomponents,sizeof(Real));

	// init classes for analysis and saving
	Safe safe(p);

	// save initial condition, ic=0
	translateArrayOrder(c,ctemp,p,untranslatedFlag,0);
	safe.save(ctemp,0);

	// prepare GPU
	init_GPU(&s,p,&h,&d,m);

	// time loop
	for(size_t step=0; step<=p.stepsEnd; step++){
		dynamics(&d,p,&s,step,m);

		// output: save state and do analysis
		if( !(step%p.stepsSaveState) and step>p.stepsSaveStateOffset ){

			// copy
			copy_GPU_to_GPU(&d,p,&s);
			copy_GPU_to_CPU(&d,c,p,&s);
			if(c[0]!=c[0]){ printf("step: %zu, c[0]=%f. Abort!\n",step,c[0]); exit(EXIT_FAILURE); }

			// translate array from concentration major to space major
			translateArrayOrder(c,ctemp,p,untranslatedFlag,0);

			// save
			if(!(step%p.stepsSaveState) and step>p.stepsSaveStateOffset){
				if(untranslatedFlag){ safe.save(c,step); }
				else{ safe.save(ctemp,step); }
			}

		}

		checkCUDAError("loop iteration",__LINE__);						// DEBUG
	}

	// clean up data
	cleanup_GPU(c,&d,p);
	free(ctemp);
}



void solver(Real *c, Real *het, params &p, modelparams &m){

	// serial version
	switch(p.spaceDim){
		case 1:														// cartesian 1d
			solverGPU_1d(c,het,p,m);
			break;
		case 1000:													// network
			solverGPU_network(c,het,p,m);
			break;
		default: printf("Error: spaceDim not implemented!\n"); exit(EXIT_FAILURE); break;
	}
}
