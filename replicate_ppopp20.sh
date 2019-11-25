#!/bin/bash

### Downloading dataset (If the data folder and 
### sunflower8w80000.points.bin are created, it skips downloading the dataset.)
if [ ! -d "data/" ] && [! -f "data/sunflower8w80000.points.bin"]; then
 mkdir ./data
 echo "Downloading point sets ..."
 wget -O ./data/Points.zip https://www.dropbox.com/sh/ab7f8gut3nh22ym/AAA0QXrC3kS0L4iHS2T0kpg-a?dl=0
 unzip ./data/Points.zip -d ./data/
 rm -f ./data/Points.zip
fi

### Run the jobs on comet or knl
cometORknl=1  # Comet is 1, KNL is 2
if [ "$#" -eq 1 ]; then
cometORknl=$1
fi


if [ $cometORknl -eq 1 ]; then
	echo "Running on Comet ... "
	sbatch matrox_comet.sh 
else
	echo "running on KNL ..."
	sbatch matrox_knl.sh
fi

echo "The generated csv and eps files are found in Figures folder"
