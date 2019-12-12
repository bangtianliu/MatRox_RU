#!/bin/bash
#SBATCH -A mit162
#SBATCH --job-name="PPoPP_artifact"
#SBATCH --output="ppopp.%j.%N.out"
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --mem=120G
#SBATCH --export=ALL
#SBATCH -t 12:00:00

module load cmake
cometORknl=1  # Comet is 1, KNL is 2
if [ "$#" -eq 1 ]; then
cometORknl=$1
fi

if [ $cometORknl -eq 1 ]; then
export MODULEPATH=/share/apps/compute/modulefiles:$MODULEPATH
module load intel/2018.1.163
module load gnu/6.2.0
module load python
module load scipy/3.6
export MKLROOT=/share/apps/compute/intel/intelmpi2018/compilers_and_libraries/linux/mkl/
export CMAKE_CXX_COMPILER=/share/apps/compute/intel/intelmpi2018/compilers_and_libraries/linux/bin/intel64/icpc
#export OMP=/share/apps/compute/intel/intelmpi2018/compilers_and_libraries/linux/lib/intel64/
fi

MatRox_HOME=$(pwd)
MatRox_build=${MatRox_HOME}/build/sympiler/
MatRox_Lib=${MatRox_HOME}/libTest/
MatRox_Fig=${MatRox_HOME}/Figures/

### Installing MatRox
mkdir build
cd build
cmake ..
cd sympiler
make

cp ../../scripts/* .
cp ../../scripts/* ${MatRox_Lib}/
# # Accuracy and Performance tests
# sbatch testMatRox 0
# sbatch testMatRox 0.03

### Figure 4: multiple right-hand-sides ###
##run MatRox
cd ${MatRox_build}
# HSS MatROx
bash nrhssh  0.0
cp hssnrhs.csv ../../Figures/Fig4-nrhs/
## H2-b MatRox
bash nrhssh 0.03
cp h2nrhs.csv ../../Figures/Fig4-nrhs/
## LibTest
cd ${MatRox_Lib}
bash GOnrhsh 0.0
cp gohssnrhs.csv ../Figures/Fig4-nrhs/
## H2b (GOFMM)
bash GOnrhsh 0.03
cp goh2nrhs.csv ../Figures/Fig4-nrhs/
##HSS (STRUMPACK)
bash stnrhsh
cp stnrhs.csv ../Figures/Fig4-nrhs/

## Figure 4-HSS
cd ${MatRox_Fig}/Fig4-nrhs/

python3 drawhssnrhs.py --m hssnrhs.csv --g gohssnrhs.csv --s stnrhs.csv

## Figure 4-H2b
python3 drawh2bnrhs.py --m h2nrhs.csv --g goh2nrhs.csv

##############################################

### Figure 5: optimization breakdown ###

cd ${MatRox_build}
## HSS MatRox
bash HSSFlops
cp hssflops.csv ../../Figures/Fig5-flops/
## H2-b MatRox
bash H2Flops
cp h2flops.csv ../../Figures/Fig5-flops/

## LibTest
cd ${MatRox_Lib}
## HSS (GOFMM)
bash testGOFlops 0.0
cp gohssflops.csv ../Figures/Fig5-flops/
## H2-b (GOFMM)
bash testGOFlops 0.03
cp goh2flops.csv ../Figures/Fig5-flops/
## HSS (STRUMPACK)
bash testSTFlops
cp stflops.csv ../Figures/Fig5-flops/

## Figure 5-HSS
cd ${MatRox_Fig}/Fig5-flops/

python3 drawhssflops.py --m hssflops.csv --g gohssflops.csv --s stflops.csv

## Figure 5-H2b
python3 drawh2bflops.py --m h2flops.csv --g goh2flops.csv

##############################################


###### Figure 7: Scalability #################

cd ${MatRox_build}
## MatRox
bash testScal
cp scal.csv ../../Figures/Fig7-scal/

## LibTest
cd ${MatRox_Lib}
## GOFMM
bash testGOScal
cp goscal.csv ../Figures/Fig7-scal/
## STRUMPACK
bash testSTScal
cp stscal.csv ../Figures/Fig7-scal/
## SMASH
bash testSMAScal
cp smascal.csv ../Figures/Fig7-scal/
cd ${MatRox_Fig}/Fig7-scal/

## Figure 7-scalability on comet
python3 drawscal.py --m scal.csv --g goscal.csv --s stscal.csv --sa smascal.csv

##############################################


###### Figure 9: Accuracy #################

cd ${MatRox_build}
bash accsh 0.03
cp acc.csv ${MatRox_Fig}/Fig9-acc/


## Figure 9 accuracy
cd ${MatRox_Fig}/Fig9-acc/
python3 drawacc.py acc.csv

##############################################


###### Figure 10: Multiple runs #################

cd ${MatRox_build}
## H2-b
bash nrunsh 0.03
cp nrun.csv ../../Figures/Fig10-nrun/
 
## LibTest
cd ${MatRox_Lib}
bash GOnrunsh 0.03
cp gonrun.csv ../Figures/Fig10-nrun/
## Figure 10-multiple runs
cd ${MatRox_Fig}/Fig10-nrun/
python3 drawnrun.py --m nrun.csv --g gonrun.csv

##############################################
