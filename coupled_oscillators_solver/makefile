
### code options
# compilers to use
CC=g++
NCC=nvcc
# add/remove for double/single precision; double is required for zbke
PRECISIONFLAG = -DDOUBLE
# add/remove for yes/no to DEBUG info
PPFLAGS = -DDEBUG
# add/remove for yes/no to filament interpolation
#~ FILAMENT_INTERPOLATION = -DFILAMENT_LEAST_SQUARES_INTERPOLATION -llapacke
# GPU or CPU ? without/with comment is GPU/CPU
#~ HARDWARE_TYPE=GPU
# initialize variables for later
name=$(shell basename "$(CURDIR)").exe
HOST=$(shell hostname)


### set variables for different hardware types and compile
ifeq ($(HARDWARE_TYPE),GPU)


## GPU

LDFLAGS=-L/usr/local/cuda/lib64
CCFLAGS = -Wall -std=c++11 -lboost_system -lboost_filesystem -fopenmp -lcublas
NCCFLAGS = -std=c++11 -lcublas
PPFLAGS += -DGPU

MAIN_SOURCES = $(wildcard ./*.cpp)
GPU_SOURCES = $(wildcard ./GPU/*.cu)
UTILITY_SOURCES = $(wildcard ./Utilities/*.cpp)
OBJECTS = $(MAIN_SOURCES:.cpp=.o) $(UTILITY_SOURCES:.cpp=.o) $(GPU_SOURCES:.cu=.o)

## set compute capability
# custom
ifeq ($(HOST),Fermion)
	# pc
	cc=52
else ifeq ($(HOST),Tachyon)
	# laptop
	cc=50
endif
# general
ifeq ($(cc),)
	SMS = 20 30 35 37 50 52 60
	$(foreach sm,$(SMS),$(eval GPUARCH += -gencode arch=compute_$(sm),code=sm_$(sm)))
else
	GPUARCH = -gencode arch=compute_$(cc),code=sm_$(cc)
endif

release: CCFLAGS += -O3 $(PRECISIONFLAG) $(FILAMENT_INTERPOLATION)
release: NCCFLAGS += -O3 $(PRECISIONFLAG)
release: all

# -use_fast_math must not be used for zbke2k!, -DDOUBLE must be used for zbke2k
fast: CCFLAGS += -O3
fast: NCCFLAGS += -O3 -use_fast_math
fast: all

# no optimization
no: CCFLAGS += $(PRECISIONFLAG)
no: NCCFLAGS += $(PRECISIONFLAG) -prec-div=true -prec-sqrt=true
no: all

debug: CCFLAGS += -g -O0 $(PRECISIONFLAG)
debug: NCCFLAGS += -g -G $(PRECISIONFLAG)
debug: all

### compilation rules
all: message $(OBJECTS)
	$(NCC) $(GPUARCH) $(NCCFLAGS) -dlink $(OBJECTS) -o gpu_final.o
	$(CC) gpu_final.o $(OBJECTS) $(CCFLAGS) -I/usr/local/cuda/include -lcudart -lboost_program_options -o $(name) $(LDFLAGS)

%.o: %.cu
	$(NCC) -x cu $(PPFLAGS) $(NCCFLAGS) $(GPUARCH) -dc $< -o $@

%.o: %.cpp
	$(CC) $(PPFLAGS) $(CCFLAGS) -c $< -o $@

message: 
	@echo
	@echo ==== building GPU version ====
	@echo

clean:
	for ext in "o" "u" "exe" "gch"; do find . -name \*.$${ext} -type f -delete; done


else

## CPU

CCFLAGS = -Wall -std=c++11 -lboost_system -lboost_filesystem -fopenmp


MAIN_SOURCES = $(wildcard ./*.cpp)
CPU_SOURCES = $(wildcard ./CPU/*.cpp)
UTILITY_SOURCES = $(wildcard ./Utilities/*.cpp)
OBJECTS = $(MAIN_SOURCES:.cpp=.o) $(CPU_SOURCES:.cpp=.o) $(UTILITY_SOURCES:.cpp=.o)

release: CCFLAGS += -O3 $(PRECISIONFLAG) $(FILAMENT_INTERPOLATION)
release: all

# -use_fast_math must not be used for zbke2k!, -DDOUBLE must be used for zbke2k
fast: CCFLAGS += -O3
fast: all

# no optimization
no: CCFLAGS += $(PRECISIONFLAG)
no: all

# valgrind --leak-check=full ./coupled_oscillators_solver.exe --ini ../ini/phase_global.ini 2> valgrindoutput.txt
debug: CCFLAGS += -g -O0 $(PRECISIONFLAG)
debug: all

### compilation rules
all: message $(OBJECTS)
	$(CC) $(OBJECTS) $(CCFLAGS) -lboost_program_options -o $(name) $(LDFLAGS)

%.o: %.cpp
	$(CC) $(PPFLAGS) $(CCFLAGS) -c $< -o $@
	
message:
	@echo
	@echo ==== building CPU version ====
	@echo
	
clean:
	for ext in "o" "exe"; do find . -name \*.$${ext} -type f -delete; done

endif
