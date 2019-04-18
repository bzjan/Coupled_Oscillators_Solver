#ifndef __solver_cpu_hpp__
#define __solver_cpu_hpp__

struct host_pointers {
	// arrays
	Real *c0;							// concentration history
	Real *c;							// current state
	Real *cnew;							// pointer to next value
	Real *cdelay;						// pointer to delay values
	Real *het;							// heterogeneity (array1)
	Real *mask;							// nonlocal convolution kernel/mask
};

#endif
