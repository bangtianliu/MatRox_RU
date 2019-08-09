# MatRox
MatRox is a code generator for generating HMatrix-matrix multiplication code, built on top of Sympiler.

## Installation

### Library requirements
MatRox does need external library (MKL) for building and testing the
MatRox-generated code, Intel MKL libraries are required.


###Setup
Modify the CMakeLists.txt in MatRox_RU/sympiler/ and MatRox_RU/matroxTest-V1/ to Modify the path to MKL library and OPENMP.

Take Comet as an example for setting the path.
```bash
set(CMAKE_CXX_COMPILER /share/apps/compute/intel/intelmpi2018/compilers_and_libraries/linux/bin/intel64/icpc)
set(OMP /share/apps/compute/intel/intelmpi2018/compilers_and_libraries/linux/lib/intel64/)

set(MKL_INCLUDE_DIRS ${MKL_ROOT}/include/)
set(MKL_LIBRARIES ${MKL_ROOT}/lib/intel64/)
```
As local machine, you need to define the path to library in CMakeLists.txt in MatRox/sympiler/ and MatRox/matroxTest-V1/ for MatRox compilation and Testing MatRox-generated code.


### Building Sympiler
```bash
cd where/you/cloned/Sympiler
mkdir build
cd build
cmake ..
cd sympiler
make
```

### Testing MatRox-generated code
The first step is to set the environmental variables corresponding
to each library. The following shows how the variables are set in bash.

```bash
set(CMAKE_CXX_COMPILER /share/apps/compute/intel/intelmpi2018/compilers_and_libraries/linux/bin/intel64/icpc)
set(OMP /share/apps/compute/intel/intelmpi2018/compilers_and_libraries/linux/lib/intel64/)

set(MKL_INCLUDE_DIRS ${MKL_ROOT}/include/)
set(MKL_LIBRARIES ${MKL_ROOT}/lib/intel64/)
After setting the library paths:

```
This will build the two remaining parts of the project including
Sympiler tests for both Cholesky and Triangular solve.

### Matrix dataset
We put the points we used in MatRox_RU/data folder.

### Evaluating MatRox
After build is done successfully, the following commands can be used
to evaluate MatRox and MatRox-generated code by running the following scripts:

#### Show the performance and accuracy
```bash
cd build/sympiler/
#HSS on comet
sbatch testMatRox 0  
#HSS local
bash testMatRox 0
#H2-b on comet
sbatch testMatRox 0.03
#H2-b local
bash testMatRox 0.03
```
The result cound be found in \*.csv

There are 6 columns:
1. tree&interaction&sample: time in tree construction, compute interaction and sampling
2. cogen: code generation time
3. sa: structure analysis
4. approximation: low-rank approximation
5. evaluation
6. acc

Compression time = item 1 + item 4


#### Show the input accuracy vs overall accuracy
```bash
cd build/sympiler/
#=
#HSS on comet
sbatch accsh 0  
#HSS local
bash accsh 0
#H2-b on comet
sbatch accsh 0.03
#H2-b local
bash accsh 0.03
```

#### Show the scalability result
```bash
#=
#HSS on comet
sbatch accsh 0  
#HSS local
bash accsh 0
#H2-b on comet
sbatch accsh 0.03
#H2-b local
bash accsh 0.03

```



## Source Tree Description

### scripts
The scripts to download the matrices used in the paper.

### symGen
The Sympiler-generated code is stored here by default and is used for testing
Sympiler in symTest folder.

### sympiler
This folder contains the source of Sympiler. Running this code generates code
for a specific sparsity.

### symTest
This folder tests the Sympiler-generated code for a given matrix.
