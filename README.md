# MatRox
MatRox is a code generator for generating efficient parallel HMatrix-matrix multiplication code, built on top of Sympiler.



## PPOPP'20 Artifact Description
The facilitate the replication of graphs in the PPoPP paper, some scripts are provided that will generate graphs of the paper on the two testbed architectures i.e., Hasewell and KNL nodes. Since the memory requirement for running all datasets are high and it is hard to find a local machine with that specification,  We provide login information to XSEDE Comet and Stampede2 servers to ensure all reviewers can run all datasets.

To reproduce graphs on Hasewell which include figures 4, 5, 9, and 10:
```
ssh XXXX@comet.sdsc.edu
#enter the provided password
git clone https://github.com/kobeliu85/MatRox_RU.git 
cd MatRox_RU
bash replicate_ppopp20.sh 1

```

To replicate the scalability figure on a KNL node, follow instructions below:
```
ssh XXXX@stampede2.tacc.utexas.edu
#enter the provided password
#enter the provided 6-digit code
cd $WORK
git clone https://github.com/kobeliu85/MatRox_RU.git 
cd MatRox_RU
bash replicate_ppopp20.sh 2

```

After the script is finished, all graphs are available under $HOME/MatRox_RU/build/sympiler/ and $WORK/MatRox_RU/build/sympiler/ in order in Comet and Stampede2 servers. The replicate_ppopp20.sh script installs MatRox, downloads datasets, generates graph data, and finally plot graphs. All steps are commented inside the script. The instruction for plotting each graph separately is available in the Figures folder.  

Similar trend should be visible in other architectures. General installation and running instructions are explained as following.  

## Installation

### Software dependencies
MatRox requires following dependencies:
* Intel MKL BLAS: for building the MatRox generated code.
* Intel C++ compiler: For compiling MatRox and also MatRox generated code.


### Setup
Modify the CMakeLists.txt in MatRox_RU/sympiler/ and MatRox_RU/matroxTest-V1/ to Modify the path to MKL library and OPENMP.

Take Comet as an example for setting the path.
```bash
set(CMAKE_CXX_COMPILER /share/apps/compute/intel/intelmpi2018/compilers_and_libraries/linux/bin/intel64/icpc)
set(OMP /share/apps/compute/intel/intelmpi2018/compilers_and_libraries/linux/lib/intel64/)

set(MKL_INCLUDE_DIRS ${MKL_ROOT}/include/)
set(MKL_LIBRARIES ${MKL_ROOT}/lib/intel64/)
```
As local machine, you need to define the path to library in CMakeLists.txt in MatRox/sympiler/, MatRox/matroxTest-V1/ and MatRox/codeTest for MatRox compilation and Testing MatRox-generated code.


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
Copy all scripts in this folder to MatRox_RU/build/sympiler/ to run the tests. You should go to MatRox_RU/build/sympiler/ to run the scripts
In the KNL subfolder, there are scipts for testing MatRox, GOFMM and SMASH on KNL.

The scripts for testing GOFMM, STRUMPACK and SMASH are named with letters "GO", "ST" and "SMA". The scripts for evaluating reference tools should be copied to libTest folder.

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

GOFMM and STRUMPACK can be evaluated by using
```bash
#GOFMM for HSS
sbatch testGOFMM 0
#STRUMPACK
sbatch testST
```

#### Show the result for multiple right hand sides (Figure 4)

```bash
#HSS on comet
sbatch nrhssh  0.0
#H2-b on comet
sbatch nrhssh 0.03
```
The results could be found in rhs1.csv, rhs1k.csv, rhs2k.csv and rhs4k.csv for W with 1, 1K, 2K and 4K columns.
The first column is compression time, second column is code generation time, third column is structure analysis time, last column is the evaluation time.

The results for GOFMM and STRUMPACK can be tested using GOnrhssh and STnrhssh

#### Show the result for optimization breakdown (Figure 5)
```bash
#HSS on comet
sbatch HSSFlops
#H2-b on comet
sbatch H2Flops
```
The result can be found in hssflops.csv and h2flops.csv respectively.
For HSS case, the columns are flops result for sequential, coarsening and low-level transformation;
For H2-b case, the columns are flops result for sequential, blocking, coarsening and low-level transformation.

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
sbatch testScal
```
The result could be found in scal.csv, in which each row has 12 items show the performance of evaluation code related to 1-12 threads.


The results for GOFMM, STRUMPACK and SMASH can be tested using scripts: testGOScal, testSTScal and testSMAScal.

#### Show the results for multiple runs (Figure 10)
```bash
#HSS on comet
sbatch nrunsh 0.0
#H2-b
sbatch nrunsh 0.03
```
The result could be found in nrun.csv

The results for GOFMM can be tested using GOnrunsh

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

### codeGen
This folder contains the generated codes for optimization breakdown.

### codeTest
This folder tests code from codeGen folder for separating the effect of different optimization techniques.

### libTest
This folder contains the binary files compiled using other reference libraries.   

### Figures
This folder contains the python scripts and instructions for generating the experiment results in the paper submission.
The subfolder refResult includes the results (csv files) and images (eps files) I generated first based on the instructions as reference.
