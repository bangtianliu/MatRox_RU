# MatRox
MatRox is a code generator for generating efficient parallel HMatrix-matrix multiplication code, built on top of Sympiler.


## PPOPP'20 Artifact Description
### Software and Hardware dependencies
MatRox requires Intel C++ compiler and Intel MKL library as software dependencies.
Meeting software dependencies, MatRox can run in any architecture. However, for running all datasets in the PPoPP20 paper, a machine with at least 40GB of RAM is required. 

The facilitate the replication of graphs in the PPoPP paper, some scripts are provided that will generate graphs of the paper on the two testbed architectures i.e., Hasewell and KNL nodes. Since the memory requirement for running all datasets are high and it is hard to find a local machine with that specification,  We provide login information to XSEDE Comet and Stampede2 servers to ensure all reviewers can run all datasets conveniently.

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


### Building MatRox
Export following environment variables:
```bash
export CMAKE_CXX_COMPILER=path/to/icpc
export MKLROOT=/path/to/mkl/lib/
```

To build the MatRox following steps have to be done:
```bash
cd where/you/cloned/MatRox
mkdir build
cd build
cmake ..
cd sympiler
make
```

## Testing MatRox-generated code
### Downloading the dataset
Download all points from this [Link](https://www.dropbox.com/sh/ab7f8gut3nh22ym/AAA0QXrC3kS0L4iHS2T0kpg-a?dl=0), and unzip it in  MatRox_RU/data folder (the folder should be created first). Assuming you are in MatRox_RU/build/sympiler, alternatively, you can use following commands to download and extract the dataset:
```bash
mkdir ../../data
wget -O ../../data/Points.zip https://www.dropbox.com/sh/ab7f8gut3nh22ym/AAA0QXrC3kS0L4iHS2T0kpg-a?dl=0
unzip ../../data/Points.zip -d ../../data/
```

Since some of used dataset is too large for local machine, we also provide 5 small datasets, i.e., TODO: XX, XX, XX, XX, and XX, which can be executed locally.

### Evaluating MatRox
After build is done successfully, the following scripts can be used
to evaluate MatRox and MatRox-generated code for given datasets. 

#### Show the performance and accuracy
```bash
cd build/sympiler/
#HSS local: example command 2
bash testMatRox 0
#H2-b local: example command 4
bash testMatRox 0.03
```
The result could be found in res.csv

There are 6 columns:
1. tree&interaction&sample: total time of tree construction, compute interaction, and sampling
2. cogen: code generation time
3. sa: structure analysis time
4. approximation: low-rank approximation time
5. evaluation: evaluation code time
6. acc: overall accuracy

Compression time = item 1 + item 4

A set of scripts are provided in scripts folder that allows exploring different strengths of MatRox. More information are available in [MatRox paper](http://www.paramathic.com/wp-content/uploads/2019/11/matrox_PPOPP.pdf).


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
