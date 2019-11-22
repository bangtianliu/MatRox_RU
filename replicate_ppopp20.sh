#!/bin/bash
# TODO: Are these the same in KNL?
module load cmake
module load intel
module load python
cometORknl=1  # Comet is 1, KNL is 2
if [ "$#" -eq 1 ]; then
cometORknl=$1
fi

if [ $cometORknl -eq 1 ]; then
export MODULEPATH=/share/apps/compute/modulefiles:$MODULEPATH
module load intel/2018.1.163 
export MKLROOT=/share/apps/compute/intel/intelmpi2018/compilers_and_libraries/linux/mkl/
export CMAKE_CXX_COMPILER=/share/apps/compute/intel/intelmpi2018/compilers_and_libraries/linux/bin/intel64/icpc
#export OMP=/share/apps/compute/intel/intelmpi2018/compilers_and_libraries/linux/lib/intel64/
fi
if [ $cometORknl -eq 2 ]; then
module load intel/18.0.2
export MKLROOT=/opt/intel/compilers_and_libraries_2018.2.199/linux/mkl/  #TODO
export CMAKE_CXX_COMPILER=/opt/intel/compilers_and_libraries_2018.2.199/linux/bin/intel64/icpc #TODO
#export OMP=/share/apps/compute/intel/intelmpi2018/compilers_and_libraries/linux/lib/intel64/
fi

### Installing MatRox
mkdir build
cd build
cmake ..
cd sympiler
make

cp ../../scripts/* .

### Downloading dataset
mkdir ../../data
wget -O ../../data/Points.zip https://www.dropbox.com/sh/ab7f8gut3nh22ym/AAA0QXrC3kS0L4iHS2T0kpg-a?dl=0
unzip ../../data/Points.zip -d ../../data/
#rm -f ../../data/Points.zip


### Data generation
if [ $cometORknl -eq 1 ]; then
# # Accuracy and Performance tests
# sbatch testMatRox 0 
# sbatch testMatRox 0.03

### Figure 4: multiple right-hand-sides
#HSS on comet
sbatch nrhssh  0.0
#H2-b on comet
sbatch nrhssh 0.03

### Figure 5: optimization breakdown
#HSS on comet
sbatch HSSFlops
#H2-b on comet
sbatch H2Flops

### Figure 9:
# #HSS on comet: example command 1
# sbatch accsh 0  
# #HSS local: example command 2
# bash accsh 0
#H2-b on comet: example command 3
sbatch accsh 0.03
# #H2-b local: example command 4
# bash accsh 0.03

### Figure 7: Scalability 
#HSS on comet
#sbatch testScal

### Figure 10: multiple runs
# #HSS on comet
# sbatch nrunsh 0.0
#H2-b
sbatch nrunsh 0.03

fi

if [ $cometORknl -eq 2 ]; then
sbatch testScalKNL
fi

### Library tests
cd ../../libTest/
if [ $cometORknl -eq 1 ]; then
# # Accuracy and Performance tests
# #GOFMM for HSS
# sbatch testGOFMM 0
# #STRUMPACK
# sbatch testST

# Figure 4
#HSS (GOFMM)
sbatch GOnrhssh 0.0
#H2b (GOFMM)
sbatch GOnrhssh 0.03
#HSS (STRUMPACK)
sbatch stnrhsh

#Figure 5
#HSS  (GOFMM)
sbatch testGOFlops 0.0
#H2-b (GOFMM)
sbatch testGOFlops 0.03

#HSS (STRUMPACK)
sbatch testSTFlops

#Figure 7
sbatch testGOScal
sbatch testSTScal
sbatch testSMAScal


# Figure 10
sbatch GOnrunsh 0.03
fi

if [ $cometORknl -eq 2 ]; then
# Figure 7
sbatch testGOScal
sbatch testSTScal 
fi

### Plotting graphs
if [ $cometORknl -eq 1 ]; then
#Figure 4-HSS
python drawhssnrhs.py --s hssnrhs.csv --g gohssnrhs.csv --t stnrhs.csv

#Figure 4-H2b
python drawh2bnrhs.py --s h2nrhs.csv --g goh2nrhs.csv

#Figure 5-HSS
python drawhssflops.py --s hssflops.csv --g gohssflops.csv --t stflops.csv

#Figure 5-H2b
python drawh2bflops.py --s h2flops.csv --g goh2flops.csv

#Figure 7
python drawscal.py --m scal.csv --g goscal.csv --s stscal.csv --sa smascal.csv

#Figure 9
python drawacc.py acc.csv

#Figure 10
python drawnrun.py --m nrun.csv --g gonrun.csv

fi

if [ $cometORknl -eq 2 ]; then
python drawscalknl.py --m scalknl.csv --g goscalknl.csv --s stscalknl.csv
fi
