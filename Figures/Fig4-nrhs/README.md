## How to generate Figure 4
#### Example for HSS
```bash
python3 drawhssnrhs.py --m [path/to/hssnrhs.csv] --g [path/to/gohssnrhs.csv] --s [path/to/stnrhs.csv]
```
- hssnrhs.csv -> running "sbatch nrhssh 0.0" under /MatRox_RU/build/sympiler/,
- gohssnrhs.csv -> running "sbatch GOnrhsh 0.0" under /MatRox_RU/libTest/ for GOFMM result.
- stnrhs..csv  -> running "sbatch stnrhsh" under /MatRox_RU/libTest for STRUMPACK result
#### Example for H2b
```bash
python3 drawh2bnrhs.py --m [path/to/h2nrhs.csv] --g [path/to/goh2nrhs.csv]
```
- h2nrhs.csv -> running "sbatch nrhssh 0.03" under /MatRox_RU/build/sympiler/,
- goh2nrhs.csv -> running "sbatch GOnrhsh 0.03" under /MatRox_RU/libTest/ for GOFMM result.

use "python3 draw*nrhs.py -h" to check the useage
