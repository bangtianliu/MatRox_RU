## How to generate Figure 7
#### Scalability result on comet
```bash
python drawscal.py --m [path/to/scal.csv] --g [path/to/goscal.csv] --s[path/to/stscal.csv] --sa[path/to/smascal.csv]
```
- scal.csv -> running "sbatch testScal" under /MatRox_RU/build/sympiler/
- goscal.csv -> running "sbatch testGOScal" under /MatRox_RU/libTest/ for GOFMM result
- stscal.csv -> running "sbatch testSTScal" under /MatRox_RU/libTest/ for STRUMPACK result
- smascal.csv -> running "sbatch testSMAScal" under /MatRox_RU/libTest/ for SMASH result

#### Scalability result on KNL

```bash
python drawscalknl.py --m [path/to/scal.csv] --g [path/to/goscal.csv] --s[path/to/stscal.csv]
```
- scalknl.csv -> running "sbatch testScalKNL" under /MatRox_RU/build/sympiler/
- goscalknl.csv -> running "sbatch testGOScalKNL" under /MatRox_RU/libTest/ for GOFMM result
- stscalknl.csv -> running "sbatch testSTScalKNL" under /MatRox_RU/libTest/ for STRUMPACK result

use "python drawscal.py -h" to check the useage
