import sys
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
import argparse

def main(filename, gfilename):
# filename = '/home/labuser/Dropbox/H2MATRICES/PPOPP/matrox/hsseps.csv'

    # with open(filename, newline='') as csvfile:
    #     data = list(csv.reader(csvfile))
    data= pd.read_csv(filename, sep=' ', header=None)
    gdata = pd.read_csv(gfilename, sep=' ', header=None)

    filterdata = 'covtype,\thiggs,\tmnist,\tsusy,\tletter,\trandom,\t,flower,\t'
    data.ix[:,0] = data.ix[:,0].map(lambda x:x.lstrip(filterdata) )
    gdata.ix[:,0] = gdata.ix[:,0].map(lambda x:x.lstrip(filterdata) )
    # print(gdata)
    # print(data)
    a = np.array(data)
    a = a.astype(np.float)

    g = np.array(gdata)
    g = g.astype(np.float)

    matseq = a[:,1]/1e+9
    matblk = (a[:,3]-a[:,1])/1e+9
    matcoar = (a[:,5]-a[:,3])/1e+9
    matlow = (a[:,7]-a[:,5])/1e+9

    print(matseq)
    goseq = a[:,0]/g[:,0]
    goseq = goseq/1e+9
    gopar = a[:,0]/g[:,1]
    gopar = gopar/1e+9
    # print(goseq)
    # matlow = a[2,:]
    # goseq = a[3,:]
    # gocoar = a[4,:]
    # strseq = a[5,:]
    # strpar = a[6,:]
    #
    bar_width = 0.5
    epsilon = .015
    line_width = 1
    opacity = 0.7

    data=['covtype','higgs', 'mnist', 'susy', 'letter', 'pen', 'hepmass', 'gas', 'grid', 'random', 'dino', 'sunflower', 'unit']

    pos_bar_positions = np.arange(len(matseq))*2
    neg_bar_positions = pos_bar_positions + bar_width

    mat_seq_bar = plt.bar(pos_bar_positions, matseq, bar_width,
                                  color='#3399FF',
                                  edgecolor='black',
                                  linewidth=line_width,
                                  label='MatRox: CDS(seq) ')

    mat_blk_bar = plt.bar(pos_bar_positions, matblk, bar_width,
                                  bottom=matseq,
                                  color='#00CCCC',
                                  edgecolor='black',
                                  linewidth=line_width,
                                  label='MatRox: CDS + block ')

    mat_coar__bar = plt.bar(pos_bar_positions, matcoar, bar_width,
                                  bottom=matseq+matblk,
                                  alpha=opacity,
                                  color='#006633',
                                  edgecolor='black',
                                  linewidth=line_width,
    #                               hatch='//',
                                  label='MatRox: CDS + block+ coarsen')
    mat_low_bar = plt.bar(pos_bar_positions, matlow, bar_width,
                         bottom=matseq+matcoar+matblk,
                          alpha=opacity,
                               color='purple',
                                   edgecolor='black',
                                   linewidth=line_width,
                                   hatch='0',
                                   label='MatRox: CDS + block + coarsen + low-level')
    go_seq_bar = plt.bar(neg_bar_positions, goseq, bar_width,
                                  color='#CC6600',
                                  edgecolor='black',
                                   linewidth=line_width,
                                  label='GOFMM: TB(seq)')
    go_par_bar = plt.bar(neg_bar_positions, gopar, bar_width,
                                  bottom=goseq,
                                  color="#FF9933",
    #                               hatch='//',
                                  edgecolor='black',
                                  linewidth=line_width,
                                  label='GOFMM: TB + DS')

#
#     plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.legend(frameon=False)
    plt.ylim(0,800)
    plt.xticks((neg_bar_positions+pos_bar_positions)/2, data, rotation=45, fontsize=7)
    plt.ylabel('GFLOPS/s')
    plt.savefig('h2-separate.eps', format='eps')
    plt.show()
#     print(matseq)
#     print(matcoar)
if __name__ == '__main__':
    # filename = '/home/labuser/Dropbox/H2MATRICES/PPOPP/matrox/hsseps.csv'
#     filename=sys.argv[1]
#     ''
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", help="require the path to hssflops.csv from matrox. if csv file does not exists, run sbatch H2Flops", default=' ')
    parser.add_argument("--g",help="require the path to gohssflops.csv from gofmm. if csv file does not exists, run sbatch testGOFlops 0.03", default='')
    args = parser.parse_args()
    main(args.m, args.g)