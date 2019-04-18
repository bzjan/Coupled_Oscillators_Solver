
# Coupled Oscillators Solver

#### Contact information
Jan Totz,  <jantotz@itp.tu-berlin.de>


###  Requirements

* Linux Ubuntu    >= 18.04
* Boost           >= 1.67
* CUDA            >= 9.2
* python          >= 3.6
* GNU make        >= 4.1
* gcc             >= 7.3.0



###  Installation  

#### 1) install boost
* download the latest version from: http://www.boost.org/users/download/  
* unzip tar.bz2 in $HOME/source:  
```
tar xvjf ~/Downloads/boost*.tar.bz2 -C ~/source
cd ~/source/boost*
./bootstrap.sh
sudo ./b2 install
```

#### 2) CUDA installation
(only required if you want to use GPU-accelerated code, otherwise skip to 3)  
* get .run file(s) (possible patches) from cuda repository https://developer.nvidia.com/cuda-toolkit  
* save both in ~/cuda  
```
chmod u+x ~/cuda/*
sudo service lightdm stop
sudo ./cuda_*.run --silent --override --toolkit --samples
```

#### 3) python installation
```
sudo apt-get install python3 python3-pip
sudo -H pip3 install --upgrade pip
sudo -H pip3 install jupyter numpy matplotlib
```

#### 4) program
for the CPU version:
```
cd coupled_oscillators_solver
# adapt line 13 in the makefile for CPU or GPU version by commenting it
make
```


##  Usage

#### 1) prepare and run simulation
* modify <nameOfIni>.ini file in ini directory to your liking  
* run program via
```
./coupled_oscillators_solver.exe --ini ../ini/<nameOfIni>.ini
```

* for example:
```
./SWC_CPU_solver.exe --ini ../ini/phase_global.ini
```

#### 2) to run the visulization, enter on the commandline:
```
jupyter notebook
```
and navigate to the directory with the .ipynb file  
* activate cells with shift+enter  
* enter name and continue
