#!/bin/bash

### Downloading dataset (If you have download the dataset, this part can be commented out)
mkdir ./data
wget -O ./data/Points.zip https://www.dropbox.com/sh/ab7f8gut3nh22ym/AAA0QXrC3kS0L4iHS2T0kpg-a?dl=0
unzip ./data/Points.zip -d ./data/
rm -f ./data/Points.zip


### Intalling MatRox (It can be commented out after installing MatRox)
mkdir build
cd build
cmake ..
cd sympiler
make


### Run the jobs on comet or knl
cometORknl=1  # Comet is 1, KNL is 2
if [ "$#" -eq 1 ]; then
cometORknl=$1
fi

cd ../../

if [ $cometORknl -eq 1 ]; then
	sbatch matrox_comet.sh 
else
	sbatch matrox_knl.sh
fi

echo "The generated csv and eps files are found in Figures folder"
