import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import sys
import re
import pandas as pd
import argparse

def main(filename):
    x=np.arange(1,14,1.0)
    data=['covtype','higgs', 'mnist', 'susy', 'letter', 'pen', 'hepmass', 'gas', 'grid', 'random', 'dino', 'sunflower', 'unit']

    adata= pd.read_csv(filename, sep=' ', header=None)
    filterdata='covtype,\thiggs,\tmnist,\tsusy,\tletter,\trandom,\t,flower,\t'
#     data.ix[:,0]
    adata.ix[:,0] = adata.ix[:,0].map(lambda x:x.lstrip(filterdata) )
#     print(adata.ix[:,0])
    adata=adata.as_matrix()
    adata = adata.astype(np.float)
    print(adata)
    e1=adata[:,0]
    e2=adata[:,1]
    e3=adata[:,2]
    e4=adata[:,3]
    e5=adata[:,4]

    plt.figure()
    plt.plot(x,e1,marker="*",label="bacc=1e-1")
    plt.plot(x,e2,marker="+",label="bacc=1e-2")
    plt.plot(x,e3,marker="o",label="bacc=1e-3")
    plt.plot(x,e4,marker="x",label="bacc=1e-4")
    plt.plot(x,e5,marker="s",label="bacc=1e-5")
    plt.xticks(x,data,rotation=45,fontsize=7)
    plt.yscale('log')
    plt.ylabel('Overall accuracy')
    plt.axhline(y=1e-1,linestyle='--',linewidth=0.5)
    plt.axhline(y=1e-2,linestyle='--',linewidth=0.5)
    plt.axhline(y=1e-3,linestyle='--',linewidth=0.5)
    plt.axhline(y=1e-4,linestyle='--',linewidth=0.5)
    plt.axhline(y=1e-5,linestyle='--',linewidth=0.5)
    plt.axhline(y=1e-6,linestyle='--',linewidth=0.5)
    plt.axhline(y=1e-7,linestyle='--',linewidth=0.5)
    plt.legend(frameon=False)
    plt.savefig('acc.eps', format='eps')
    plt.show()

if __name__ == '__main__':
#     filename = '/home/labuser/Dropbox/H2MATRICES/PPOPP/matrox/acc.csv'
    # filename='/home/labuser/acc.csv'
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="require the path to acc.csv. if acc.csv does not exists, run sbatch accsh 0.03 ")
    args = parser.parse_args()
    # filename=sys.argv[1]
#     ''
    main(args.filename)