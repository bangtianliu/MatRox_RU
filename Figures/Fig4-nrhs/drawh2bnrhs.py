import sys
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse
import pandas as pd

def main(filename, gfilename):

    data = pd.read_csv(filename, sep=',', header=None)
    gdata = pd.read_csv(gfilename, sep='\t', header=None)


    filterdata = 'covtype,\thiggs,\tmnist,\tsusy,\tletter,\trandom,\t,flower,\t'
    data.ix[:,0] = data.ix[:,0].map(lambda x:x.lstrip(filterdata) )
    gdata.ix[:, 0] = gdata.ix[:, 0].map(lambda x: x.lstrip(filterdata))
    gdata.ix[:,2] =gdata.ix[:,2].map(lambda x: x.rstrip(','))



    # data.ix[:, 1] = data.ix[:, 1].map(lambda x: x.lstrip(filterdata))
    # data.ix[:, 2] = data.ix[:, 2].map(lambda x: x.lstrip(filterdata))
    data=data.ix[:, 1:4]
    data= data.as_matrix()
    data = data.astype(np.float)

    gdata=gdata.ix[:,1:2]
    gdata=gdata.as_matrix()


    g = np.array(gdata)
    g = g.astype(np.float)

    indx = np.array([0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15])
    # indy = np.array([0,1,2,3])
    # print(ind)
    a=data[indx,:]
    g=g[indx,:]
 #   print(g)
 #   print(data[indx, :])

    a = a.transpose()
    g = g.transpose()
    #
    # matcom = a[0, [0, 4, 8, 12]]
    # print(matcom)
    matcom = a[0,:]
    matsa = a[1, :]
    matcg = a[2, :]
    mateval = a[3, :]
    gocom = g[0, :]
    goeval = g[1, :]
    # strcom = a[6, :]
    # streval = a[7, :]
    #
    bar_width = 0.5
    epsilon = .015
    line_width = 1
    opacity = 0.7
    #
    pos_bar_positions = np.arange(len(matcom)) * 2
    neg_bar_positions = pos_bar_positions + bar_width

    data = ['higgs-1', 'higgs-1K', 'higgs-2K', 'higgs-4K',
            'susy-1', 'susy-1K', 'susy-2K', 'susy-4K',
            'letter-1', 'letter-1K', 'letter-2K', 'letter-4K',
            'grid-1', 'grid-1K', 'grid-2K', 'grid-4K']
    mat_com_bar = plt.bar(pos_bar_positions, matcom, bar_width,
                          color='#E0E0E0',
                          edgecolor='black',
                          linewidth=line_width,
                          label='MatRox-compression ')
    mat_sa_bar = plt.bar(pos_bar_positions, matsa, bar_width,
                         bottom=matcom,
                         color='#CC6600',
                         edgecolor='black',
                         linewidth=line_width,
                         label='MatRox-structure analysis ')

    mat_cg_bar = plt.bar(pos_bar_positions, matcg, bar_width,
                         bottom=matcom + matsa,
                         alpha=opacity,
                         color='#FF9933',
                         edgecolor='black',
                         linewidth=line_width,
                         #                               hatch='//',
                         label='MatRox-code generation')

    mat_eval_bar = plt.bar(pos_bar_positions, mateval, bar_width,
                           bottom=matcom + matsa + matcg,
                           alpha=opacity,
                           color='purple',
                           edgecolor='black',
                           linewidth=line_width,
                           hatch='0',
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

    #
    # #     plt.legend(bbox_to_anchor=(1.6, 1.05))
    plt.legend(frameon=False, fontsize=14)
    plt.xlim(-2, 34)
    # plt.ylim(0, 40)
    plt.xticks((neg_bar_positions + pos_bar_positions) / 2, data, rotation=45, fontsize=7)
    plt.ylabel('Time (seconds)')
    plt.savefig('h2nrhs.eps', format='eps')
    #plt.show()
    # print(a[0, :])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--m", help="require the path to hssnrhs.csv from matrox. if csv file does not exists, run sbatch nrhssh 0.03", default='./hssnrhs.csv')
    parser.add_argument("--g", help="require the path to gohssnrhs.csv from gofmm. if csv file does not exists, run sbatch GOnrhsh 0.03", default='./gohssnrhs.csv')
    # parser.add_argument("--s", help="require the path to stflops.csv from strumpack. if csv file does not exists, run sbatch stnrhsh", default='./stnrhs.csv')
    args = parser.parse_args()

    main(args.m, args.g)
