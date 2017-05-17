#!/usr/bin/env python3
import math
import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

folderprefix = "recombinationartisoutput/"

fs = 16
fig, axes = plt.subplots(4, 1, sharey=False, figsize=(6, 4*3), tight_layout={"pad": 0.3, "w_pad": 0.0, "h_pad": 0.0})

romannumerals = ('', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII',
                 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX')

Arad = [1.42e-13, 1.02e-12, 3.32e-12, 7.80e-12]
Xrad = [8.91e-01, 8.43e-01, 7.46e-01, 6.82e-01]
Adi  = [1.58e-03, 8.38e-03, 1.53e-02, 3.75e-02]
Bdi  = [4.56e-01, 3.23e-01, 3.10e-01, 4.11e-01]
T0   = [6.00e+04, 1.94e+05, 3.31e+05, 4.32e+05]
T1   = [8.97e+04, 1.71e+05, 2.73e+05, 3.49e+05]

for ionindex in range(4):
    xlist = []
    ylist = []
    xlistartisold = []
    ylistartisold = []
    ylistSS82 = []  # radiative recombination
    ylistSS82withDI = []  # radiative and dielectric recombination
    with open(folderprefix + 'recombinationartisoutput.txt', 'r') as filein:
        for line in filein:
            row = line.split()
            if row[3] == '0' and int(row[5]) == ionindex:
                T = float(row[7])
                xlist.append(math.log10(T))
                weightfactor = [1/3., 9/25., 1., 1/25.][ionindex]
                if row[9] == 'nan':
                    ylist.append(0.0)
                else:
                    ylist.append(float(row[9]) * weightfactor)

                ylistSS82.append(Arad[ionindex] * (T/1e4) ** -Xrad[ionindex])
                alphadSS82 = Adi[ionindex] * (T ** -3.0/2.0) * math.exp(-T0[ionindex]/T) * (1 + Bdi[ionindex] * math.exp(-T1[ionindex]/T))
                ylistSS82withDI.append(ylistSS82[-1] + alphadSS82)
                # print('DER Correction',alphadSS82/ylistSS82[-1])
#    print(Arad[ionindex],Bdi[ionindex],T0[ionindex],T1[ionindex])

    with open(folderprefix + 'recombinationartisoutputold.txt', 'r') as filein:
        for line in filein:
            row = line.split()
            if row[3] == '21' and int(row[5]) == ionindex:
                T = float(row[7])
                xlistartisold.append(math.log10(T))
                # weightfactor = [1/3.,9/25.,1.,1/25.][ionindex]
                weightfactor = 1.0
                ylistartisold.append(float(row[9]) * weightfactor)

    axes[ionindex].plot(xlistartisold, ylistartisold, marker='None', lw=2, label='Fe {0} ARTIS old'.format(romannumerals[ionindex+1]))
    axes[ionindex].plot(xlist, ylist, marker='None', lw=2, label='Fe {0} ARTIS Hillier'.format(romannumerals[ionindex+1]))
    axes[ionindex].set_xlim(xmin=3.5, xmax=xlist[-1])
#    axes[ionindex].set_ylim(ymin=min(ylist+ylistSS82+ylistartisold),ymax=max(ylist+ylistSS82+ylistartisold)*3)

    # axes[ionindex].plot(xlist, ylistSS82, marker='None', lw=3,
    #                    label='Fe {0} S&S1982'.format(romannumerals[ionindex+1]))
    # axes[ionindex].plot(xlist, ylistSS82withDI, marker='None', lw=1,
    #                    label='Fe {0} (w/ DER) S&S1982'.format(romannumerals[ionindex+1]))

    # ARTIS with NAHAR photoionization cross sections
    # ('recombinationartisoutput-naharptpx.txt','Fe {0} ARTIS Nahar.ptpx'.format(romannumerals[ionindex+1])),
#    for (artisoutputfilename,strlabel) in [('recombinationartisoutput-naharpx.txt','Fe {0} ARTIS Nahar.px'.format(romannumerals[ionindex+1])),('recombinationartisoutput-naharpx-togsonly.txt','Fe {0} ARTIS Nahar.(>gs).px'.format(romannumerals[ionindex+1])),('recombinationartisoutput-naharpx_gsoldcode.txt','Fe {0} ARTIS Nahar.oldcode.(>gs).px'.format(romannumerals[ionindex+1])),('recombinationartisoutput-naharpxnaharlevelsgsonly.txt','Fe {0} ARTIS Nahar.(>gs).px.Nahar.levels'.format(romannumerals[ionindex+1])),('recombinationartisoutput-naharpxnaharlevels.txt','Fe {0} ARTIS Nahar.px.Nahar.levels'.format(romannumerals[ionindex+1]))]:
    for (artisoutputfilename, strlabel) in [
            ('recombinationartisoutput-naharpx.txt', 'Fe {0} ARTIS Nahar.px'.format(romannumerals[ionindex+1])),
            ('recombinationartisoutput-naharpxnaharlevels.txt', 'Fe {0} ARTIS Nahar.px.Nahar.levels'.format(romannumerals[ionindex+1])),
            ('recombinationartisoutput_hilliernaharcombined.txt', 'Fe {0} ARTIS Nahar.px.NaharHillier.levels'.format(romannumerals[ionindex+1])),
            ('recombinationartisoutput-Oct2016.txt', 'Fe {0} ARTIS Oct 2016'.format(romannumerals[ionindex+1]))]:
        xlist = []
        ylist = []
        with open(folderprefix + artisoutputfilename, 'r') as filein:
            for line in filein:
                row = line.split()
                if line.startswith('Alpha result:') and row[3] == '0' and int(row[5]) == ionindex:
                    T = float(row[7])
                    xlist.append(math.log10(T))
                    # weightfactor = [1/3.,9/25.,1.,1/25.][ionindex]
                    weightfactor = 1.0
                    ylist.append(float(row[9]) * weightfactor)
        axes[ionindex].plot(xlist, ylist, marker='None', lw=2, label=strlabel)


def naharfeiitonumber(strin):
    return float(strin[:4]) * 10 ** (-float(strin[6:8]))

xlist = []
ylist = []
with open(folderprefix + 'recombinationdatanahar97fei.txt', 'r') as filein:
    for line in filein:
        row = line.split()
        xlist.append(float(row[0]))
        ylist.append(float('E-'.join(row[2].split('-'))))
# print(list(zip(xlist,ylist)))
axes[0].plot(xlist, ylist, marker='None', lw=2, label='Fe I Nahar (1997)')

for (filename, label, ax) in [
        ['recombinationdatanahar97feii.txt', 'Fe II Nahar (1997)', axes[1]],
        ['recombinationdatanahar96feiii.txt', 'Fe III Nahar (1996)', axes[2]],
        ['recombinationdatanahar98feiv.txt', 'Fe IV Nahar (1998)', axes[3]]]:
    xlist = []
    ylist = []
    with open(folderprefix + filename, 'r') as filein:
        for line in filein:
            row = line.split()
            xlist.append(float(row[0]))
            ylist.append(naharfeiitonumber(row[1]))
        filein.seek(0)
        for line in filein:
            row = line.split()
            xlist.append(float(row[2]))
            ylist.append(naharfeiitonumber(row[3]))
        filein.seek(0)
        for line in filein:
            row = line.split()
            if len(row) >= 6:
                xlist.append(float(row[4]))
                ylist.append(naharfeiitonumber(row[5]))
    # print(list(zip(xlist,ylist)))
    ax.plot(xlist, ylist, marker="None", lw=2, label=label)

for ax in axes:
    ax.legend(loc='best', ncol=2, handlelength=1.5, frameon=False, numpoints=1, prop={'size': fs-8})

    # ax.xaxis.set_major_locator(ticker.MultipleLocator(base=1.0))
    # ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=0.2))
#    ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=0.01))
#    plt.setp(plt.getp(ax, 'xticklabels'), fontsize=fsticklabel)
#    plt.setp(plt.getp(ax, 'yticklabels'), fontsize=fsticklabel)
#    for axis in ['top','bottom','left','right']:
#        ax.spines[axis].set_linewidth(framewidth)
#    ax.annotate(modellabel, xy=(0.97, 0.95), xycoords='axes fraction', horizontalalignment='right', verticalalignment='top', fontsize=fs)
    ax.set_yscale('log')
    ax.set_ylabel(r'Alpha [cm$^3$ s$^{-1}$]', fontsize=fs)
axes[-1].set_xlabel(r'Log$_{10}$ T', fontsize=fs)
fig.savefig(__file__ + '.pdf', format='pdf')
plt.close()
