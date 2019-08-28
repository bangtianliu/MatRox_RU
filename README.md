# MatRox
MatRox is a code generator for generating HMatrix-matrix multiplication code, built on top of Sympiler.

## Installation

### Library requirements
MatRox does need external library (MKL) for building and testing the
MatRox-generated code, Intel MKL libraries are required.


### Setup
Modify the CMakeLists.txt in MatRox_RU/sympiler/ and MatRox_RU/matroxTest-V1/ to Modify the path to MKL library and OPENMP.

Take Comet as an example for setting the path.
```bash
set(CMAKE_CXX_COMPILER /share/apps/compute/intel/intelmpi2018/compilers_and_libraries/linux/bin/intel64/icpc)
set(OMP /share/apps/compute/intel/intelmpi2018/compilers_and_libraries/linux/lib/intel64/)

set(MKL_INCLUDE_DIRS ${MKL_ROOT}/include/)
set(MKL_LIBRARIES ${MKL_ROOT}/lib/intel64/)
```
As local machine, you need to define the path to library in CMakeLists.txt in MatRox/sympiler/ and MatRox/matroxTest-V1/ for MatRox compilation and Testing MatRox-generated code.


### Building MatRox
```bash
cd where/you/cloned/MatRox
mkdir build
cd build
cmake ..
cd sympiler
make
```
### Scripts
Please copy all scripts in this folder to MatRox_RU/build/sympiler/ to run the tests. You should go to MatRox_RU/build/sympiler/ to run the scripts

### Testing MatRox-generated code
The first step is to set the environmental variables corresponding
to each library. The following shows how the variables are set in bash.

On comet, use the following commands to load intel compiler
```bash
export MODULEPATH=/share/apps/compute/modulefiles:$MODULEPATH
module load intel/2018.1.163
```

```bash
set(CMAKE_CXX_COMPILER /share/apps/compute/intel/intelmpi2018/compilers_and_libraries/linux/bin/intel64/icpc)
set(OMP /share/apps/compute/intel/intelmpi2018/compilers_and_libraries/linux/lib/intel64/)

set(MKL_INCLUDE_DIRS ${MKL_ROOT}/include/)
set(MKL_LIBRARIES ${MKL_ROOT}/lib/intel64/)
```

### dataset

[Link to obtain Points](https://www.dropbox.com/sh/ab7f8gut3nh22ym/AAA0QXrC3kS0L4iHS2T0kpg-a?dl=0), Select all and
download the dataset in folder named points
Please put the dataset directly to MatRox_RU/data folder.You can check the
scripts to make sure you put the dataset into correct directories.
Since some of used dataset is too large for local machine, we also provide 5 small datasets, which can be executed locally.

### Evaluating MatRox
After build is done successfully, the following commands can be used
to evaluate MatRox and MatRox-generated code by running the following scripts:

#### Show the performance and accuracy
```bash
cd build/sympiler/
#HSS on comet: example command 1
sbatch testMatRox 0  
#HSS local: example command 2
bash testMatRox 0
#H2-b on comet: example command 3
sbatch testMatRox 0.03
#H2-b local: example command 4
bash testMatRox 0.03
```
The result could be found in res.csv

There are 6 columns:
1. tree&interaction&sample: time in tree construction, compute interaction and sampling
2. cogen: code generation time
3. sa: structure analysis
4. approximation: low-rank approximation
5. evaluation: the time for evaluation code
6. acc: overal accuracy

Compression time = item 1 + item 4

#### Show the result for multiple right hand sides (Figure 4)

```bash
#HSS on comet
sbatch nrhssh  0.0
#H2-b on comet
sbatch nrhssh 0.03
```
The results could be found in rhs1.csv, rhs1k.csv, rhs2k.csv and rhs4k.csv for W with 1, 1K, 2K and 4K columns.
The first column is compression time, second column is code generation time, third column is structure analysis time, last column is the evaluation time.

#### Show the input accuracy vs overall accuracy (Figure 9)
```bash
cd build/sympiler/
#=
#HSS on comet: example command 1
sbatch accsh 0  
#HSS local: example command 2
bash accsh 0
#H2-b on comet: example command 3
sbatch accsh 0.03
#H2-b local: example command 4
bash accsh 0.03
```
The result could be found in acc.csv, in which each row has 5 items show the overall accuracies related to 1e-1,1e-2,1e-3,1e-4 and 1e-5 for one dataset.


#### Show the scalability result (Figure 7)
```bash
#=
#HSS on comet
sbatch testScal 0  
#HSS local
bash testScal 0
#H2-b on comet
sbatch testScal 0.03
#H2-b local
bash testScal 0.03

```
The result could be found in scal.csv, in which each row has 12 items show the performance of evaluation code related to 1-12 threads.

#### Show the results for multiple runs (Figure 10)
```bash
#HSS on comet
sbatch nrunsh 0.0
#H2-b
sbatch nrunsh 0.03
```
The result could be found in nrun.csv

## Source Tree Description

### scripts
The scripts to run the code, and should be copied to build/sympiler/ folder.

### symGen
The MatRox-generated code is stored here by default and is used for testing
Sympiler in matroxTest-V1 folder.

### sympiler
This folder contains the source of MatRox based on sympiler code. Running this code generates code
for a specific structure.

### matroxTest-V1
This folder tests the MatRox-generated code for a given structure, in which low-rank approximation is performed at run-time.
