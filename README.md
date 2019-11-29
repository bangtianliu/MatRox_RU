# MatRox
MatRox is a code generator for generating efficient parallel HMatrix-matrix multiplication code, built on top of Sympiler.

The Readme file is created in two sections, artifact for PPoPP'20 paper titled MatRox: Modular approach for improving data locality in Hierarchical (Mat)rix App(Rox)imation, and general instructions for installing MatRox in a generic architecture. 

## 1. PPOPP'20 Artifact Description
### Software and Hardware dependencies
MatRox requires Intel C++ compiler and Intel MKL library as software dependencies.
Meeting software dependencies, MatRox can run in any architecture. However, for running all datasets in the PPoPP20 paper, a machine with at least 40GB of RAM is required. 

### Replicating Graphs
To facilitate the replication of graphs of the PPoPP paper, two scripts are provided that will generate graphs of the paper on the two testbed architectures i.e., Hasewell and KNL nodes. Since the memory requirement for running all datasets is more than 40GB, and it is hard to find a local machine with that requirement, we setup MatRox code on two XSEDE servers, i.e., Comet and Stampede2. 
For Library codes, binaries are provided for all libraries, i.e., GOFMM, STRUMPACK, and SMASH. The source code and the driver code for GOFMM and STRUMPACK are included in the libTest folder as two separate zip files. The SMASH code is not public, so only its binary is provided. 

 **Note to reviewers:** if you do not have access to XSEDE servers, we can provide temporary login information to XSEDE Comet and Stampede2 servers to ensure you can run all datasets conveniently. Please coordinate this through Hotcrp forum. 

To reproduce graphs on Hasewell which include figures 4, 5, 7 (upper part), 9, and 10:
```
ssh username@comet.sdsc.edu
#enter the provided password
git clone https://github.com/kobeliu85/MatRox_RU.git 
cd MatRox_RU
bash replicate_ppopp20.sh 1

```

To replicate the scalability figure, i.e., lower part of figure 7, on a KNL node, follow instructions below:
```
ssh username@stampede2.tacc.utexas.edu
#enter the provided password 
#enter the provided 6-digit code
cd $WORK
git clone https://github.com/kobeliu85/MatRox_RU.git 
cd MatRox_RU
bash replicate_ppopp20.sh 2

```

After the script is finished, all graphs are available under $HOME/MatRox_RU/Figures/ ain order in Comet. Figures are stored under the directory of the figure number. For Stampede2 server, generated csv files are under the $WORK/MatRox_RU/Figures/Fig7-scal/ directory. The user should copy csv files into a local machine with python and follow  instructions in lines 61-64 of matrox_knl.sh .
The replicate_ppopp20.sh script installs MatRox, downloads datasets, generates graph data, and finally plot graphs. All steps are commented inside the script. 
The instruction for plotting each graph separately is also available in the Figures folder for interested reviewers.  

Similar trend should be visible in other architectures. General installation and running instructions are explained as following.  

## 2. General Installation

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

Since some of used dataset is too large for local machine, we also provide 5 small datasets, i.e., (covtype, mnist, SUSY, HIGGS, HEPMASS)16384.points.bin, which can be executed locally (using testsmallpoints script).

### Evaluating MatRox
After build is done successfully, the following scripts can be used
to evaluate MatRox and MatRox-generated code for given datasets. 

#### Show the performance and accuracy
```bash
cd build/sympiler/
#HSS : 
bash testMatRox 0
#H2-b : you will need a minimum of 40 GB to run this script
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
This folder contains the source of MatRox based on the Sympiler code. This project generates code
for a specific structure.

### matroxTest-V1
This folder tests the MatRox-generated code for a given structure, in which low-rank approximation is performed at run-time.

### codeGen
This folder contains the MatRox-generated code for optimization breakdown.

### codeTest
This folder tests code from codeGen folder for separating the effect of different optimization techniques.

### libTest
This folder contains the binary files of other reference libraries.   

### Figures
This folder contains the python scripts and instructions for generating the experiment results in the MatRox paper.
The subfolder refResult provides results (csv files) and images (eps files) for sanity check.
