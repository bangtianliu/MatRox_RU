import sys
import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse
import pandas as pd

def main(filename, gfilename):
    data = pd.read_csv(filename, sep=',')
    gdata = pd.read_csv(gfilename, sep=',', header=None)


    filterdata = 'covtype,\thiggs,\tmnist,\tsusy,\tletter,\trandom,\t,flower,\t'
    data.ix[:,0] = data.ix[:,0].map(lambda x:x.lstrip(filterdata) )
    gdata.ix[:, 0] = gdata.ix[:, 0].map(lambda x: x.lstrip(filterdata))
    gdata = gdata.as_matrix()
    gdata = gdata[:,1:3]
    print(gdata)

    # gdata.ix[:,2] =gdata.ix[:,2].map(lambda x: x.rstrip(','))
    data = data.ix[:,1:4]
    print(data)
    print(data.as_matrix())
    a = np.array(data)
    a = a.astype(np.float)
    print(a.sum(axis=1))
    total= a.sum(axis=1)
    a = a.transpose()
    matinp1 = a[0, :]/total
    matinp2 = a[1, :]/total
    matexe = a[2, :]/total

    g = np.array(gdata)
    g = g.astype(np.float)
    g = g.transpose()


    gocom = g[0, :]/total
    goeval = g[1, :]/total

    bar_width = 0.5
    epsilon = .015
    line_width = 1
    opacity = 0.7

    data = ['covtype', 'higgs', 'mnist', 'susy', 'letter', 'pen', 'hepmass', 'gas', 'grid', 'random', 'dino',
            'sunflower', 'unit']

    pos_bar_positions = np.arange(len(matexe)) * 2
    neg_bar_positions = pos_bar_positions + bar_width

    mat_inp1_bar = plt.bar(pos_bar_positions, matinp1, bar_width,
                           color='#E0E0E0',
                           edgecolor='black',
                           linewidth=line_width,
                           label='MatRox-inspector-p1 ')

    mat_inp2_bar = plt.bar(pos_bar_positions, matinp2, bar_width,
                           bottom=matinp1,
                           color='#FFFFFF',
                           edgecolor='black',
                           linewidth=line_width,
                           label='MatRox-inspector-p2 ')

    mat_exe__bar = plt.bar(pos_bar_positions, matexe, bar_width,
                           bottom=matinp1 + matinp2,
                           alpha=opacity,
                           color='purple',
                           edgecolor='black',
                           linewidth=line_width,
                           #                               hatch='//',
                           label='MatRox-executor')
    go_com_bar = plt.bar(neg_bar_positions, gocom, bar_width,
                         color='#606060',
                         edgecolor='black',
                         linewidth=line_width,
                         label='GOFMM-compression')
    go_eval_bar = plt.bar(neg_bar_positions, goeval, bar_width,
                          bottom=gocom,
                          color="#99CCFF",
                          #                               hatch='//',
                          edgecolor='black',
                          linewidth=line_width,
                          label='GOFMM-evaluation')
    #     plt.legend(bbox_to_anchor=(1.6, 1.05))
    plt.legend(frameon=False)
    plt.xticks((neg_bar_positions + pos_bar_positions) / 2, data, rotation=45, fontsize=7)
    plt.ylabel('Normalized execution time')
    plt.ylim([0, 4.0])
    plt.savefig('matnrun.eps', format='eps')
    plt.show()
    print(matinp2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", help="require the path to nrun.csv from matrox. if csv file does not exists, run sbatch nrunsh 0.03", default='./nrun.csv')
    parser.add_argument("--g", help="require the path to gonrun.csv from gofmm. if csv file does not exists, run sbatch GOnunsh 0.03", default='./gonrun.csv')
    # parser.add_argument("--s", help="require the path to stflops.csv from strumpack. if csv file does not exists, run sbatch stnrhsh", default='./stnrhs.csv')
    args = parser.parse_args()

    main(args.m, args.g)