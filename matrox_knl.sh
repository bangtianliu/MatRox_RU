#!/bin/bash

#SBATCH -J "PPoPP"
#SBATCH -o "PPoPP.o%j"
#SBATCH -e "PPoPP.e%j"
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -A TG-CCR180004


module cmake
module load intel/18.0.2
export MKLROOT=/opt/intel/compilers_and_libraries_2018.2.199/linux/mkl/ 
export CMAKE_CXX_COMPILER=/opt/intel/compilers_and_libraries_2018.2.199/linux/bin/intel64/icpc


MatRox_HOME=$(pwd)
MatRox_build=${MatRox_HOME}/build/sympiler/
MatRox_Lib=${MatRox_HOME}/libTest/
MatRox_Fig=${MatRox_HOME}/Figures/



### Installing MatRox
#mkdir build
cd build
cmake ..
cd sympiler
make


cp ../../scripts/KNL/testScalKNL ./
# Figure 7
bash testScalKNL
cp scalknl.csv ../../Figures/Fig7-scal/
cd ../../libTest/
bash testGOScal
cp goscalknl.csv ../../Figures/Fig7-scal/
bash testSTScal
cp stscalknl.csv ../../Figures/Fig7-scal/


### the csv files should be copied to local machine and using follow command to draw figures. 
echo "please copy figures to local machine and draw figures"
# cd ../Figures/Fig7-scal/
# python drawscalknl.py --m scalknl.csv --g goscalknl.csv --s stscalknl.csv

