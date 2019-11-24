import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse
import pandas as pd

def main(filename, gfilename, sfilename):
    data = pd.read_csv(filename, sep=' |,', header=None)
    gdata = pd.read_csv(gfilename, sep=' |,', header=None)
    sdata = pd.read_csv(sfilename, sep=' ', header=None)

    print(gdata)

    filterdata = 'covtype,\thiggs,\tmnist,\tsusy,\tletter,\trandom,\t,flower,\t-h2\t'
    # data.ix[:,0] = data.ix[:,0].map(lambda x:x.lstrip(filterdata) )
    data = data.ix[:,1:13]
    gdata = gdata.ix[:,1:13]


    sdata.ix[:,0] = sdata.ix[:,0].map(lambda x:x.lstrip(filterdata) )
    print(sdata)

    a = np.array(data)
    a = a.astype(np.float)
    b = np.array(a[:,0])
    a = b[:,np.newaxis] / a

    g = np.array(gdata)
    g = g.astype(np.float)
    seq = np.array(g[:,0])
    g = seq[:,np.newaxis] / g

    s = np.array(sdata)
    s = s.astype(np.float)
    s[0,:] = s[0,0] / s[0,:]


    #print(a)
    threads = np.array([1, 2, 4, 8, 17, 34, 68])
    plt.plot(threads, a[0,:], 'b->', label='MatRox-HSS')
    plt.plot(threads, g[0,:], 'r-o', label='GOFMM-HSS')
    plt.plot(threads, a[2,:], 'c-<', label='MatRox-$\mathcal{H}^2$-b')
    plt.plot(threads, g[2,:], 'y-+', label='GOFMM-$\mathcal{H}^2$-b')
    plt.plot(threads, threads, 'm--', label='Linear scaling')
    plt.legend(loc='best',frameon=False)
    plt.title('covtype(KNL)')
    # plt.ylim(0,12.5)
    plt.xticks(threads)
    plt.ylabel('Speedup over serial code')
    plt.xlabel('Number of cores')
    # plt.xscale('log')

    plt.savefig('knlscalcovtype.eps', format='eps')
    plt.close()
    #plt.show()
    plt.plot(threads, a[1, :], 'b->', label='MatRox-HSS')
    plt.plot(threads, g[1, :], 'r-o', label='GOFMM-HSS')
    plt.plot(threads, s[0, :], 'g-x', label='STRUMPACK-HSS')
    plt.plot(threads, a[3, :], 'c-<', label='MatRox-$\mathcal{H}^2$-b')
    plt.plot(threads, g[3, :], 'y-+', label='GOFMM-$\mathcal{H}^2$-b')
    plt.plot(threads, threads, 'm--', label='Linear scaling')
    #
    # print(a[3,:])
    #
    plt.legend(loc='best', frameon=False)
    plt.title('unit(KNL)')
    plt.ylim(0, 68)
    # plt.xscale('log')


    plt.ylabel('Speedup over serial code')
    plt.xlabel('Number of cores')
    plt.xticks(threads)

    plt.savefig('knlscalunit.eps', format='eps')
    #plt.show()
    # print(data.ix[:,0])
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--m",help="require the path to knlscal.csv from matrox. if csv file does not exists, run sbatch testScal", default='./scal.csv')
    parser.add_argument("--g",help="require the path to goknlscal.csv from gofmm. if csv file does not exists, run sbatch testGOScal", default='./goscal.csv' )
    parser.add_argument("--s",help="require the path to stknlscal.csv from strumpack. if csv file does not exists, run sbatch testSTScal", default='./stscal.csv')
    args = parser.parse_args()
    main(args.m, args.g, args.s)
