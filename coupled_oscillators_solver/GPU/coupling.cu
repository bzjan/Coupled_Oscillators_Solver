
#include <string>								// std::string (in struct of params.hpp)
#include <vector>								// std::vector (for params.hpp)

// CUDA
#include <thrust/transform_reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>


// custom
#include "../params.hpp"						// struct params
#include "./solver_gpu.hu"						// struct device_pointers
#include "./vector_types_operator_overloads.hu"





template <int BC, int mask_radius>
__global__ void nonlocal_delay_homo_tiled_zbke2k_1d(Real2 *input, Real2 *output, Real2 *input_delay, int nx, int o_tile_width,
	const Real * __restrict__ M, const Real2 diffcoeff){

	// declare shared memory arrays for tiles, BLOCK_WIDTH = TILE_WIDTH, but BLOCK_WIDTH > O_TILE_WIDTH
	extern __shared__ Real input_shared[];

	// thread indices, no dependence on blockIdx, blockDim to support tiling. 1 tile = 1 block, 1 output field < 1 block
	int tx = threadIdx.x;

	// row and column calculation, since in- and output tiles have different sizes
	int o = blockIdx.x*o_tile_width+tx;
	int i = o-mask_radius;

	// global boundary conditions
	switch(BC){
		case 0:
			// periodic
			switch(mask_radius){
				case 5:
					if(i==-5) i += nx;
					if(i==nx+4) i -= nx;
				case 4:
					if(i==-4) i += nx;
					if(i==nx+3) i -= nx;
				case 3:
					if(i==-3) i += nx;
					if(i==nx+2) i -= nx;
				case 2:
					if(i==-2) i += nx;
					if(i==nx+1) i -= nx;
				case 1:
					if(i==-1) i += nx;
					if(i==nx) i -= nx;
					break;
			}
			break;
		// neumann
		case 1:
			switch(mask_radius){
				case 5:
					if(i==-5) i += 9;
					if(i==nx+4) i -= 9;
				case 4:
					if(i==-4) i += 7;
					if(i==nx+3) i -= 7;
				case 3:
					if(i==-3) i += 5;
					if(i==nx+2) i -= 5;
				case 2:
					if(i==-2) i += 3;
					if(i==nx+1) i -= 3;
				case 1:
					if(i==-1) i++;
					if(i==nx) i--;
					break;
			}
			break;
	}

	// all threads in one block load global memory data into shared memory to initialize shared memory for later use
	input_shared[tx] = input_delay[i].y;
	__syncthreads();

	// calculation, not all threads are needed. Threads at tile boundaries are excluded.
	Real output_temp{};
	if(tx<o_tile_width){
		Real input0=input[o].y;
		for(int i=0; i<2*mask_radius+1; i++){
			output_temp += M[i]*(input_shared[i+tx]-input0);
		}
	}
	__syncthreads();

	// write output, exclude output from threads, which contributed to loading data into shared memory but did not calc output
	//~ if(o<nx && tx<o_tile_width) output[o] += diffcoeff*max(output_temp,-1.1e-4f);
	if(o<nx && tx<o_tile_width) output[o] += diffcoeff*output_temp;
}






void nonlocal_delay_homo_tiled_zbke2k(device_pointers *d, params &p, streams *s){

	//~ float coeff=diffcoeff[0];
	//~ printf("coeff: %f\n",coeff);
	//~ printf("coeff: %f, %p || %p\n",coeff,u,unew);

	//~ printf("coeffs: %f,%f\n",d->coupling_coeffs2.x,d->coupling_coeffs2.y);
	//~ printf("mask0: %f\n",d->mask[0]);
	//~ exit(1);

	int mem_size=sizeof(Real);

	int maskWidth=2*p.cutoffRange+1;
	//~ int mask_radius=mask_width/2;
	int o_TileWidth=p.blockWidth-maskWidth+1;

	dim3 nblocks((p.nx-1)/o_TileWidth+1);
	dim3 nthreads(p.blockWidth);

	if(p.bc=="periodic"){
		switch(p.spaceDim){
			case 1:
			mem_size*=p.blockWidth;
				switch(p.cutoffRange){
					case 1: nonlocal_delay_homo_tiled_zbke2k_1d<0,1><<<nblocks,nthreads,mem_size,s->stream1>>>((Real2*)d->c,(Real2*)d->cnew,(Real2*)d->cdelay,p.nx,o_TileWidth,d->mask,d->coupling_coeffs2); break;
					case 2: nonlocal_delay_homo_tiled_zbke2k_1d<0,2><<<nblocks,nthreads,mem_size,s->stream1>>>((Real2*)d->c,(Real2*)d->cnew,(Real2*)d->cdelay,p.nx,o_TileWidth,d->mask,d->coupling_coeffs2); break;
					case 3: nonlocal_delay_homo_tiled_zbke2k_1d<0,3><<<nblocks,nthreads,mem_size,s->stream1>>>((Real2*)d->c,(Real2*)d->cnew,(Real2*)d->cdelay,p.nx,o_TileWidth,d->mask,d->coupling_coeffs2); break;
					case 4: nonlocal_delay_homo_tiled_zbke2k_1d<0,4><<<nblocks,nthreads,mem_size,s->stream1>>>((Real2*)d->c,(Real2*)d->cnew,(Real2*)d->cdelay,p.nx,o_TileWidth,d->mask,d->coupling_coeffs2); break;
					case 5: nonlocal_delay_homo_tiled_zbke2k_1d<0,5><<<nblocks,nthreads,mem_size,s->stream1>>>((Real2*)d->c,(Real2*)d->cnew,(Real2*)d->cdelay,p.nx,o_TileWidth,d->mask,d->coupling_coeffs2); break;
					default: printf("cutoffRange must be < 6! Program Abort!"); exit(1); break;
				}
				break;
			default:
				printf("spaceDim is not chosen correctly for diffusion! Program Abort!");
				exit(1);
				break;
		}
	}else if(p.bc=="neumann"){
		//~ printf("neumann\n");
		switch(p.spaceDim){
			case 1:
				mem_size*=p.blockWidth;
				switch(p.cutoffRange){
					case 1: nonlocal_delay_homo_tiled_zbke2k_1d<1,1><<<nblocks,nthreads,mem_size,s->stream1>>>((Real2*)d->c,(Real2*)d->cnew,(Real2*)d->cdelay,p.nx,o_TileWidth,d->mask,d->coupling_coeffs2); break;
					case 2: nonlocal_delay_homo_tiled_zbke2k_1d<1,2><<<nblocks,nthreads,mem_size,s->stream1>>>((Real2*)d->c,(Real2*)d->cnew,(Real2*)d->cdelay,p.nx,o_TileWidth,d->mask,d->coupling_coeffs2); break;
					case 3: nonlocal_delay_homo_tiled_zbke2k_1d<1,3><<<nblocks,nthreads,mem_size,s->stream1>>>((Real2*)d->c,(Real2*)d->cnew,(Real2*)d->cdelay,p.nx,o_TileWidth,d->mask,d->coupling_coeffs2); break;
					case 4: nonlocal_delay_homo_tiled_zbke2k_1d<1,4><<<nblocks,nthreads,mem_size,s->stream1>>>((Real2*)d->c,(Real2*)d->cnew,(Real2*)d->cdelay,p.nx,o_TileWidth,d->mask,d->coupling_coeffs2); break;
					case 5: nonlocal_delay_homo_tiled_zbke2k_1d<1,5><<<nblocks,nthreads,mem_size,s->stream1>>>((Real2*)d->c,(Real2*)d->cnew,(Real2*)d->cdelay,p.nx,o_TileWidth,d->mask,d->coupling_coeffs2); break;
					default: printf("cutoffRange must be < 6! Program Abort!"); exit(1); break;
				}
				break;
			default: printf("spaceDim is not chosen correctly for diffusion! Program Abort!"); exit(1); break;
		}
		printf("boundary condition is not implemented! Program Abort!"); exit(1);
	}

	checkCUDAError("nonlocal_delay_homo_tiled_zbke2k()",__LINE__);
}









template <int BC>
__global__ void nonlocal_delay_homo_zbke2k_1d(Real2 *input, Real2 *output, Real2 *delay, int kdia, int kradius, Real ksum, int nx, const Real * __restrict__ M, const Real2 couplecoeff){

	// thread indices
	int idx = threadIdx.x + blockIdx.x*blockDim.x;


	if(idx<nx){
		Real sum {};
		// global boundary conditions
		switch(BC){
			case 0:											// periodic
				for(int kx=0; kx<kdia; kx++){
					int jx=idx+kx-kradius;
					if(jx<0){ jx+=nx; } else if(jx>=nx){ jx-=nx; }
					sum += M[kx]*delay[jx].y;
				}
				sum -= ksum*input[idx].y;

				output[idx] += couplecoeff*sum;
				break;

			case 1:											// neumann
				for(int kx=0; kx<kdia; kx++){
					int jx=idx+kx-kradius;
					if(jx >= 0 && jx < nx){
						sum += M[kx]*(delay[jx].y - input[idx].y);
					}
				}

				output[idx] += couplecoeff*sum;
				break;
		}
	}
}


void nonlocal_delay_homo_zbke2k(device_pointers *d, params &p, streams *s){

	dim3 nblocks((p.nx-1)/p.blockWidth+1);
	dim3 nthreads(p.blockWidth);

	if(p.bc=="periodic"){
		switch(p.spaceDim){
			case 1: nonlocal_delay_homo_zbke2k_1d<0><<<nblocks,nthreads>>>((Real2*)d->c,(Real2*)d->cnew,(Real2*)d->cdelay,p.kdia,p.kradius,p.ksum,p.nx,d->mask,d->coupling_coeffs2); break;
			default: printf("spaceDim is not chosen correctly for nonlocal_delay_homo_zbke2k! Program Abort!"); exit(1); break;
		}
	}else if(p.bc=="neumann"){
		switch(p.spaceDim){
			case 1: nonlocal_delay_homo_zbke2k_1d<1><<<nblocks,nthreads>>>((Real2*)d->c,(Real2*)d->cnew,(Real2*)d->cdelay,p.kdia,p.kradius,p.ksum,p.nx,d->mask,d->coupling_coeffs2); break;
			default: printf("spaceDim is not chosen correctly for nonlocal_delay_homo_zbke2k! Program Abort!"); exit(1); break;
		}
	}else{
		printf("boundary condition is not implemented! Program Abort!"); exit(1);
	}

	checkCUDAError("nonlocal_delay_homo_zbke2k()",__LINE__);
}






template <int BC, int mask_radius>
__global__ void nonlocal_homo_tiled_zbke2k_1d(Real2* input, Real2* output, int nx, int o_tile_width, const Real* __restrict__ M, const Real2 diffcoeff){

	// declare shared memory arrays for tiles, BLOCK_WIDTH = TILE_WIDTH, but BLOCK_WIDTH > O_TILE_WIDTH
	extern __shared__ Real input_shared[];

	// thread indices, no dependence on blockIdx, blockDim to support tiling. 1 tile = 1 block, 1 output field < 1 block
	int tx = threadIdx.x;

	// row and column calculation, since in- and output tiles have different sizes
	int o = blockIdx.x*o_tile_width+tx;
	int i = o-mask_radius;

	// global boundary conditions
	switch(BC){
		case 0:
			// periodic
			switch(mask_radius){
				case 5:
					if(i==-5) i += nx;
					if(i==nx+4) i -= nx;
				case 4:
					if(i==-4) i += nx;
					if(i==nx+3) i -= nx;
				case 3:
					if(i==-3) i += nx;
					if(i==nx+2) i -= nx;
				case 2:
					if(i==-2) i += nx;
					if(i==nx+1) i -= nx;
				case 1:
					if(i==-1) i += nx;
					if(i==nx) i -= nx;
					break;
			}
			break;
		// neumann
		case 1:
			switch(mask_radius){
				case 5:
					if(i==-5) i += 9;
					if(i==nx+4) i -= 9;
				case 4:
					if(i==-4) i += 7;
					if(i==nx+3) i -= 7;
				case 3:
					if(i==-3) i += 5;
					if(i==nx+2) i -= 5;
				case 2:
					if(i==-2) i += 3;
					if(i==nx+1) i -= 3;
				case 1:
					if(i==-1) i++;
					if(i==nx) i--;
					break;
			}
			break;
	}

	input_shared[tx] = input[i].y;
	__syncthreads();

	// calculation, not all threads are needed. Threads at tile boundaries are excluded.
	// one thread performs all multiplications + additions for one output element
	Real output_temp{};
	if(tx<o_tile_width){
		Real input0=input_shared[tx+mask_radius];
		for(int i=0; i<2*mask_radius+1; i++){
			output_temp += M[i]*(input_shared[i+tx]-input0);
		}
	}
	__syncthreads();

	// write output, exclude output from threads, which contributed to loading data into shared memory but did not calc output
	//~ if(o<nx && tx<o_tile_width) output[o] += diffcoeff*max(output_temp,-1.1e-4f);
	if(o<nx && tx<o_tile_width) output[o] += diffcoeff*output_temp;

}


void nonlocal_homo_tiled_zbke2k(device_pointers *d, params &p, streams *s){

	//~ printf("nonlocal_homo_tiled_zbke2k\n");
	//~ printf("p.spaceDim: %d, p.cutoffRange: %d\n",p.spaceDim,p.cutoffRange);

	int mem_size=sizeof(Real);

	int maskWidth=2*p.cutoffRange+1;
	//~ //int mask_radius=mask_width/2;
	int o_TileWidth=p.blockWidth-maskWidth+1;


	dim3 nblocks((p.nx-1)/o_TileWidth+1);
	dim3 nthreads(p.blockWidth);

	if(p.bc=="periodic"){
		//~ printf("periodic\n");
		switch(p.spaceDim){
			case 1:
			mem_size*=p.blockWidth;
				switch(p.cutoffRange){
					case 1: nonlocal_homo_tiled_zbke2k_1d<0,1><<<nblocks,nthreads,mem_size,s->stream1>>>((Real2*)d->c,(Real2*)d->cnew,p.nx,o_TileWidth,d->mask,d->coupling_coeffs2); break;
					case 2: nonlocal_homo_tiled_zbke2k_1d<0,2><<<nblocks,nthreads,mem_size,s->stream1>>>((Real2*)d->c,(Real2*)d->cnew,p.nx,o_TileWidth,d->mask,d->coupling_coeffs2); break;
					case 3: nonlocal_homo_tiled_zbke2k_1d<0,3><<<nblocks,nthreads,mem_size,s->stream1>>>((Real2*)d->c,(Real2*)d->cnew,p.nx,o_TileWidth,d->mask,d->coupling_coeffs2); break;
					case 4: nonlocal_homo_tiled_zbke2k_1d<0,4><<<nblocks,nthreads,mem_size,s->stream1>>>((Real2*)d->c,(Real2*)d->cnew,p.nx,o_TileWidth,d->mask,d->coupling_coeffs2); break;
					case 5: nonlocal_homo_tiled_zbke2k_1d<0,5><<<nblocks,nthreads,mem_size,s->stream1>>>((Real2*)d->c,(Real2*)d->cnew,p.nx,o_TileWidth,d->mask,d->coupling_coeffs2); break;
					default: printf("cutoffRange must be < 6! Program Abort!"); exit(1); break;
				}
				break;
			default:
				printf("spaceDim is not chosen correctly for diffusion! Program Abort!");
				exit(EXIT_FAILURE);
				break;
		}
	}else if(p.bc=="neumann"){
		switch(p.spaceDim){
			case 1:
			mem_size*=p.blockWidth;
				switch(p.cutoffRange){
					case 1: nonlocal_homo_tiled_zbke2k_1d<1,1><<<nblocks,nthreads,mem_size,s->stream1>>>((Real2*)d->c,(Real2*)d->cnew,p.nx,o_TileWidth,d->mask,d->coupling_coeffs2); break;
					case 2: nonlocal_homo_tiled_zbke2k_1d<1,2><<<nblocks,nthreads,mem_size,s->stream1>>>((Real2*)d->c,(Real2*)d->cnew,p.nx,o_TileWidth,d->mask,d->coupling_coeffs2); break;
					case 3: nonlocal_homo_tiled_zbke2k_1d<1,3><<<nblocks,nthreads,mem_size,s->stream1>>>((Real2*)d->c,(Real2*)d->cnew,p.nx,o_TileWidth,d->mask,d->coupling_coeffs2); break;
					case 4: nonlocal_homo_tiled_zbke2k_1d<1,4><<<nblocks,nthreads,mem_size,s->stream1>>>((Real2*)d->c,(Real2*)d->cnew,p.nx,o_TileWidth,d->mask,d->coupling_coeffs2); break;
					case 5: nonlocal_homo_tiled_zbke2k_1d<1,5><<<nblocks,nthreads,mem_size,s->stream1>>>((Real2*)d->c,(Real2*)d->cnew,p.nx,o_TileWidth,d->mask,d->coupling_coeffs2); break;
					default: printf("cutoffRange must be < 6! Program Abort!"); exit(1); break;
				}
				break;
			default:
				printf("spaceDim is not chosen correctly for diffusion! Program Abort!");
				exit(EXIT_FAILURE);
				break;
		}
	}else{
		printf("boundary condition is not implemented! Program Abort!"); exit(1);
	}

	checkCUDAError("nonlocal_homo_tiled_zbke2k()",__LINE__);
}














template <int BC>
__global__ void nonlocal_homo_zbke2k_1d(Real2 *input, Real2 *output, int kdia, int kradius, Real ksum, int nx, const Real * __restrict__ M, const Real2 couplecoeff){

	// thread indices
	int idx = threadIdx.x + blockIdx.x*blockDim.x;

	if(idx<nx){
		Real sum {};
		// global boundary conditions
		switch(BC){
			case 0:											// periodic
				for(int kx=0; kx<kdia; kx++){
					int jx=idx+kx-kradius;
					if(jx<0){ jx+=nx; }else if(jx>=nx){ jx-=nx; }
					sum += M[kx]*input[jx].y;
				}
				sum -= ksum*input[idx].y;

				output[idx] += couplecoeff*sum;		// calculation
				break;

			case 1:											// neumann
				//~ Real ksumGPU {};
				for(int kx=0; kx<kdia; kx++){
					int jx=idx+kx-kradius;
					if(jx >= 0 && jx < nx){
						sum += M[kx]*(input[jx].y - input[idx].y);
						//~ ksumGPU += M[kx];
					}
				}
				//~ sum -= ksumGPU*input[idx].y;

				output[idx] += couplecoeff*sum;		// calculation
				break;
		}
	}
}



void nonlocal_homo_zbke2k(device_pointers *d, params &p, streams *s){


	dim3 nblocks((p.nx-1)/p.blockWidth+1);
	dim3 nthreads(p.blockWidth);

	if(p.bc=="periodic"){
		switch(p.spaceDim){
			case 1: nonlocal_homo_zbke2k_1d<0><<<nblocks,nthreads>>>((Real2*)d->c,(Real2*)d->cnew,p.kdia,p.kradius,p.ksum,p.nx,d->mask,d->coupling_coeffs2); break;
			default: printf("spaceDim is not chosen correctly for nonlocal_homo_zbke2k! Program Abort!"); exit(1); break;
		}
	}else if(p.bc=="neumann"){
		switch(p.spaceDim){
			case 1: nonlocal_homo_zbke2k_1d<1><<<nblocks,nthreads>>>((Real2*)d->c,(Real2*)d->cnew,p.kdia,p.kradius,p.ksum,p.nx,d->mask,d->coupling_coeffs2); break;
			default: printf("spaceDim is not chosen correctly for nonlocal_homo_zbke2k! Program Abort!"); exit(1); break;
		}
	}else{
		printf("boundary condition is not implemented! Program Abort!"); exit(1);
	}

	checkCUDAError("nonlocal_homo_zbke2k()",__LINE__);
}





template <int BC>
__global__ void nonlocal_sin_homo_1d(Real1* input, Real1* output, int nx, int kdia, int kradius, const Real * __restrict__ M, const Real1 couplecoeff){

	// thread indices
	int idx = threadIdx.x + blockIdx.x*blockDim.x;

	if(idx<nx){
		Real1 output_temp {};
		// global boundary conditions
		switch(BC){
			case 0:											// periodic
				for(int kx=0; kx<kdia; kx++){
					int jx=idx+kx-kradius;
					if(jx<0){ jx+=nx; } else if(jx>=nx){ jx-=nx; }
					output_temp += M[kx]*sin(input[jx].x - input[idx].x - couplecoeff.x);
				}

				output[idx] += output_temp;					// calculation
				break;

			case 1:											// neumann
				for(int kx=0; kx<kdia; kx++){
					int jx=idx+kx-kradius;
					if(jx >= 0 && jx < nx){
						output_temp += M[kx]*sin(input[jx].x - input[idx].x - couplecoeff.x);
					}
				}

				output[idx] += output_temp;					// calculation
				break;
		}
	}
}



void nonlocal_sin_homo(device_pointers *d, params &p, streams *s){

	dim3 nblocks((p.nx-1)/p.blockWidth+1);
	dim3 nthreads(p.blockWidth);

	if(p.bc=="periodic"){
		switch(p.spaceDim){
			case 1: nonlocal_sin_homo_1d<0><<<nblocks,nthreads>>>((Real1*)d->c,(Real1*)d->cnew,p.nx,p.kdia,p.kradius,d->mask,d->coupling_coeffs1); break;
			default: printf("spaceDim is not chosen correctly for nonlocal_homo! Program Abort!"); exit(1); break;
		}
	}else if(p.bc=="neumann"){
		switch(p.spaceDim){
			case 1: nonlocal_sin_homo_1d<1><<<nblocks,nthreads>>>((Real1*)d->c,(Real1*)d->cnew,p.nx,p.kdia,p.kradius,d->mask,d->coupling_coeffs1); break;
			default: printf("spaceDim is not chosen correctly for nonlocal_homo! Program Abort!"); exit(1); break;
		}
	}else{
		printf("boundary condition is not implemented! Program Abort!"); exit(1);
	}

	checkCUDAError("nonlocal_sin_homo()",__LINE__);
}









template <int BC, int mask_radius>
__global__ void nonlocal_sin_homo_tiled_1d(Real1* input, Real1* output, int nx, int o_tile_width, const Real* __restrict__ M, const Real1 diffcoeff){

	// declare shared memory arrays for tiles, BLOCK_WIDTH = TILE_WIDTH, but BLOCK_WIDTH > O_TILE_WIDTH
	extern __shared__ Real input_shared[];

	// thread indices, no dependence on blockIdx, blockDim to support tiling. 1 tile = 1 block, 1 output field < 1 block
	int tx = threadIdx.x;

	// row and column calculation, since in- and output tiles have different sizes
	int o = blockIdx.x*o_tile_width+tx;
	int i = o-mask_radius;

	// global boundary conditions
	switch(BC){
		case 0:						// periodic
			switch(mask_radius){
				case 5:
					if(i==-5) i += nx;
					if(i==nx+4) i -= nx;
				case 4:
					if(i==-4) i += nx;
					if(i==nx+3) i -= nx;
				case 3:
					if(i==-3) i += nx;
					if(i==nx+2) i -= nx;
				case 2:
					if(i==-2) i += nx;
					if(i==nx+1) i -= nx;
				case 1:
					if(i==-1) i += nx;
					if(i==nx) i -= nx;
					break;
			}
			break;
		case 1:						// neumann
			switch(mask_radius){
				case 5:
					if(i==-5) i += 9;
					if(i==nx+4) i -= 9;
				case 4:
					if(i==-4) i += 7;
					if(i==nx+3) i -= 7;
				case 3:
					if(i==-3) i += 5;
					if(i==nx+2) i -= 5;
				case 2:
					if(i==-2) i += 3;
					if(i==nx+1) i -= 3;
				case 1:
					if(i==-1) i++;
					if(i==nx) i--;
					break;
			}
			break;
	}

	input_shared[tx] = input[i].x;
	__syncthreads();

	// calculation, not all threads are needed. Threads at tile boundaries are excluded.
	// one thread performs all multiplications + additions for one output element
	Real output_temp{};
	if(tx<o_tile_width){
		Real input0=input_shared[tx+mask_radius];
		for(int i=0; i<2*mask_radius+1; i++){
			output_temp += M[i]*sin(input_shared[i+tx]-input0-diffcoeff.x);
		}
	}
	__syncthreads();

	// write output, exclude output from threads, which contributed to loading data into shared memory but did not calc output
	if(o<nx && tx<o_tile_width) output[o] += output_temp;

}




void nonlocal_sin_homo_tiled(device_pointers *d, params &p, streams *s){

	int mem_size=0;

	int maskWidth=2*p.cutoffRange+1;
	int o_TileWidth=p.blockWidth-maskWidth+1;

	dim3 nblocks((p.nx-1)/o_TileWidth+1);
	dim3 nthreads(p.blockWidth);

	//~ printf("%f\n",d->coupling_coeffs1.x);

	if(p.bc=="periodic"){
		switch(p.spaceDim){
			case 1:
				mem_size=p.blockWidth;
				switch(p.cutoffRange){
					case 1: nonlocal_sin_homo_tiled_1d<0,1><<<nblocks,nthreads,mem_size*sizeof(Real1),s->stream1>>>((Real1*)d->c,(Real1*)d->cnew,p.nx,o_TileWidth,d->mask,d->coupling_coeffs1); break;
					case 2: nonlocal_sin_homo_tiled_1d<0,2><<<nblocks,nthreads,mem_size*sizeof(Real1),s->stream1>>>((Real1*)d->c,(Real1*)d->cnew,p.nx,o_TileWidth,d->mask,d->coupling_coeffs1); break;
					case 3: nonlocal_sin_homo_tiled_1d<0,3><<<nblocks,nthreads,mem_size*sizeof(Real1),s->stream1>>>((Real1*)d->c,(Real1*)d->cnew,p.nx,o_TileWidth,d->mask,d->coupling_coeffs1); break;
					case 4: nonlocal_sin_homo_tiled_1d<0,4><<<nblocks,nthreads,mem_size*sizeof(Real1),s->stream1>>>((Real1*)d->c,(Real1*)d->cnew,p.nx,o_TileWidth,d->mask,d->coupling_coeffs1); break;
					case 5: nonlocal_sin_homo_tiled_1d<0,5><<<nblocks,nthreads,mem_size*sizeof(Real1),s->stream1>>>((Real1*)d->c,(Real1*)d->cnew,p.nx,o_TileWidth,d->mask,d->coupling_coeffs1); break;
				}
				break;
			default:
				printf("spaceDim is not chosen correctly for diffusion! Program Abort!");
				exit(EXIT_FAILURE);
				break;
		}
	}else if(p.bc=="neumann"){
		switch(p.spaceDim){
			case 1:
				mem_size=p.blockWidth;
				switch(p.cutoffRange){
					case 1: nonlocal_sin_homo_tiled_1d<1,1><<<nblocks,nthreads,mem_size*sizeof(Real1),s->stream1>>>((Real1*)d->c,(Real1*)d->cnew,p.nx,o_TileWidth,d->mask,d->coupling_coeffs1); break;
					case 2: nonlocal_sin_homo_tiled_1d<1,2><<<nblocks,nthreads,mem_size*sizeof(Real1),s->stream1>>>((Real1*)d->c,(Real1*)d->cnew,p.nx,o_TileWidth,d->mask,d->coupling_coeffs1); break;
					case 3: nonlocal_sin_homo_tiled_1d<1,3><<<nblocks,nthreads,mem_size*sizeof(Real1),s->stream1>>>((Real1*)d->c,(Real1*)d->cnew,p.nx,o_TileWidth,d->mask,d->coupling_coeffs1); break;
					case 4: nonlocal_sin_homo_tiled_1d<1,4><<<nblocks,nthreads,mem_size*sizeof(Real1),s->stream1>>>((Real1*)d->c,(Real1*)d->cnew,p.nx,o_TileWidth,d->mask,d->coupling_coeffs1); break;
					case 5: nonlocal_sin_homo_tiled_1d<1,5><<<nblocks,nthreads,mem_size*sizeof(Real1),s->stream1>>>((Real1*)d->c,(Real1*)d->cnew,p.nx,o_TileWidth,d->mask,d->coupling_coeffs1); break;
				}
				break;
			default:
				printf("spaceDim is not chosen correctly for diffusion! Program Abort!");
				exit(EXIT_FAILURE);
				break;
		}
	}else{
		printf("boundary condition is not implemented! Program Abort!"); exit(EXIT_FAILURE);
	}

	checkCUDAError("nonlocal_sin_homo_tiled()",__LINE__);
}





template <typename T>
struct sin_functor{ __host__ __device__ T operator()
	( const T& x ) const { return sin(x); }
};

template <typename T>
struct cos_functor{ __host__ __device__ T operator()
	( const T& x ) const { return cos(x); }
};



__global__ void network_global_coupling_phase_step(Real* c, Real* cnew, int len, Real multiplier, Real psi){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	
	if(i<len){
		cnew[i] += multiplier*sin(psi-c[i]);
	}
}




// global coupling via order parameter
void network_global_coupling_phase(Real* c, Real* cnew, Real* temp, params &p){
	
	// 1) get complex order parameter components
	thrust::device_ptr<Real> c_thrust(c);
	thrust::device_ptr<Real> temp_thrust(temp);
	
	thrust::copy(c_thrust, c_thrust+p.n, temp_thrust);
	Real a = thrust::transform_reduce(temp_thrust, temp_thrust+p.n, cos_functor<Real>(), 0.0, thrust::plus<Real>() ) / p.n;		// destructive operation on nw_temp
	
	thrust::copy(c_thrust, c_thrust+p.n, temp_thrust);
	Real b = thrust::transform_reduce(temp_thrust, temp_thrust+p.n, sin_functor<Real>(), 0.0, thrust::plus<Real>() ) / p.n;		// destructive operation on nw_temp
	
	Real r = sqrt(a*a + b*b);					// magnitude of order parameter
	Real psi = atan2(b,a);						// phase of order parameter
	Real multiplier = p.dt*p.K*r;
	
	// 2) coupling update: cnew_i += dt*K*r*sin(psi - phi_i)
	dim3 nblocks((p.n-1)/p.blockWidth+1);
	dim3 nthreads(p.blockWidth);
	network_global_coupling_phase_step<<<nblocks,nthreads>>>(c,cnew,p.n,multiplier,psi);
}



__global__ void network_global_coupling_frustration_phase_step(Real* c, Real* cnew, int len, Real multiplier, Real psi, Real alpha){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	
	if(i<len){
		cnew[i] += multiplier*sin(psi-c[i]-alpha);
	}
}


// global coupling via order parameter
void network_global_coupling_frustration_phase(Real* c, Real* cnew, Real* temp, params &p, modelparams &m){
	
	// 1) get complex order parameter components
	thrust::device_ptr<Real> c_thrust(c);
	thrust::device_ptr<Real> temp_thrust(temp);
	
	thrust::copy(c_thrust, c_thrust+p.n, temp_thrust);
	Real a = thrust::transform_reduce(temp_thrust, temp_thrust+p.n, cos_functor<Real>(), 0.0, thrust::plus<Real>() ) / p.n;		// destructive operation on nw_temp
	
	thrust::copy(c_thrust, c_thrust+p.n, temp_thrust);
	Real b = thrust::transform_reduce(temp_thrust, temp_thrust+p.n, sin_functor<Real>(), 0.0, thrust::plus<Real>() ) / p.n;		// destructive operation on nw_temp
	
	Real r = sqrt(a*a + b*b);					// magnitude of order parameter
	Real psi = atan2(b,a);						// phase of order parameter
	Real multiplier = p.dt*p.K*r;
	
	// 2) coupling update: cnew_i += dt*K*r*sin(psi - phi_i - alpha)
	dim3 nblocks((p.n-1)/p.blockWidth+1);
	dim3 nthreads(p.blockWidth);
	network_global_coupling_frustration_phase_step<<<nblocks,nthreads>>>(c,cnew,p.n,multiplier,psi,m.coupling_coeffs[0]);
	
}





// can be used for instantaneous and delayed coupling
__global__ void network_global_coupling_zbke_step(Real2* c, Real2* cnew, int len, Real2 multiplier, Real vmean){
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	
	if(i<len){
		cnew[i].x += multiplier.x*(vmean - c[i].y);
		cnew[i].y += multiplier.y*(vmean - c[i].y);
	}
}




void network_global_coupling_delay_zbke(Real2* c, Real2* cnew, Real2* cdelay, Real2* temp, params &p, modelparams &m){
	
	// 1) calculate delayed mean value <v(t-\tau)>
	thrust::device_ptr<Real2> cdelay_thrust(cdelay);
	thrust::device_ptr<Real2> temp_thrust(temp);
	
	thrust::copy(cdelay_thrust, cdelay_thrust+p.n, temp_thrust);
	Real2 init {};
	Real vmean = thrust::reduce(cdelay_thrust,cdelay_thrust+p.n,init).y / p.n;
	
	// 2) coupling update: cnew_i += dt*K*(ku,kv)*( <v(t-\tau)> - v_i(t) )
	Real2 multipliers{p.dt*p.K*m.coupling_coeffs[0], p.dt*p.K*m.coupling_coeffs[1]};
	dim3 nblocks((p.n-1)/p.blockWidth+1);
	dim3 nthreads(p.blockWidth);
	network_global_coupling_zbke_step<<<nblocks,nthreads>>>(c,cnew,p.n,multipliers,vmean);
}






void network_global_coupling_zbke(Real2* c, Real2* cnew, Real2* temp, params &p, modelparams &m){
	
	// 1) calculate mean value <v(t)>
	thrust::device_ptr<Real2> c_thrust(c);
	thrust::device_ptr<Real2> temp_thrust(temp);
	
	thrust::copy(c_thrust, c_thrust+p.n, temp_thrust);
	Real2 init {};
	Real vmean = thrust::reduce(c_thrust,c_thrust+p.n,init).y / p.n;
	
	// 2) coupling update: cnew_i += dt*K*(ku,kv)*( <v(t)> - v_i(t) )
	Real2 multipliers{p.dt*p.K*m.coupling_coeffs[0], p.dt*p.K*m.coupling_coeffs[1]};
	dim3 nblocks((p.n-1)/p.blockWidth+1);
	dim3 nthreads(p.blockWidth);
	network_global_coupling_zbke_step<<<nblocks,nthreads>>>(c,cnew,p.n,multipliers,vmean);
}





void coupling(device_pointers *d, params &p, streams *s, size_t step, modelparams &m){
	
	// iterate over all components by changing the offset
	// move offset in array to address different components
	if(step>=p.stepsCouplingStart){
		switch(p.couplingChoice){
			
			case 5: 											// nonlocal kernel, convolution
				if(p.reactionModel==24 or p.reactionModel==2401){
					if(p.use_tiles){ nonlocal_homo_tiled_zbke2k(d,p,s); }
					else{ nonlocal_homo_zbke2k(d,p,s); }
				}else if(p.reactionModel==16 or p.reactionModel==1601){
					if(p.use_tiles){ nonlocal_sin_homo_tiled(d,p,s); }
					else{ nonlocal_sin_homo(d,p,s); }
				}else{
					printf("coupling not implemented!\n"); exit(EXIT_FAILURE);
				}
				break;
			
			
			case 6:												// nonlocal, delay; fill history until delaySteps+1: only local dynamics, no coupling
				
				if(step<p.delayStartSteps+1){									// before delay history is full
					if(p.reactionModel==24 or p.reactionModel==2401){
						if(p.use_tiles){ nonlocal_homo_tiled_zbke2k(d,p,s); }
						else{ nonlocal_homo_zbke2k(d,p,s); }
					}else if(p.reactionModel==16 or p.reactionModel==1601){
						if(p.use_tiles){ nonlocal_sin_homo_tiled(d,p,s); }
						else{ nonlocal_sin_homo(d,p,s); }
					}else{
						printf("coupling not implemented!\n"); exit(EXIT_FAILURE);
					}
				}else if(step>=p.delayStartSteps+1){							// after delay history is full
					if(p.reactionModel==24 or p.reactionModel==2401){
						if(p.use_tiles){ nonlocal_delay_homo_tiled_zbke2k(d,p,s); }
						else{ nonlocal_delay_homo_zbke2k(d,p,s); }
					}else{
						printf("coupling not implemented!\n"); exit(EXIT_FAILURE);
					}
				}
				break;
			
			
			case 1002:												// global network
				if(p.reactionModel==24 or p.reactionModel==2401){
					network_global_coupling_zbke((Real2*)d->c,(Real2*)d->cnew,p,m);
				}else if(p.reactionModel==16 or p.reactionModel==1601){
					network_global_coupling_phase(d->c,d->cnew,d->nw_temp,p);
				}
				break;
			
			
			case 1003:												// global network with frustration
				if(p.reactionModel==16 or p.reactionModel==1601){
					network_global_coupling_frustration_phase(d->c,d->cnew,d->nw_temp,p,m);
				}else{
					printf("Error: coupling not implemented! (coupling @ line: %d)\n",__LINE__); exit(EXIT_FAILURE);
				}
				break;
			
			
			case 1004:													// global network with delay; fill history until delaySteps+1: only local dynamics, no coupling
				if(step<p.delayStartSteps+1){							// before delay history is full
					
					if(p.reactionModel==24 or p.reactionModel==2401){
						network_global_coupling_zbke((Real2*)d->c,(Real2*)d->cnew,p,m);
					}else{
						printf("Error: coupling not implemented! (coupling @ line: %d)\n",__LINE__); exit(EXIT_FAILURE);
					}
				}else if(step>=p.delayStartSteps+1){				// after delay history is full
					
					if(p.reactionModel==24 or p.reactionModel==2401){
						network_global_coupling_delay_zbke((Real2*)d->c,(Real2*)d->cnew,(Real2*)d->cdelay,p,m);
					}else{
						printf("Error: coupling not implemented! (coupling @ line: %d)\n",__LINE__); exit(EXIT_FAILURE);
					}
				}
				break;
			

			default: printf("Error: couplingChoice \"%d\" not implemented!\n",p.couplingChoice); exit(EXIT_FAILURE); break;
		}
	}

	checkCUDAError("coupling()",__LINE__);
}
