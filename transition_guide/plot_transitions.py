#!/usr/bin/env python3
import os
import math
# import matplotlib.ticker as ticker
from collections import namedtuple
import numpy as np
import pandas as pd

K_B = 8.617332478e-5  # eV / K
c   = 299792.458      # km / s

plot_xmin = 3500        # plot range in angstroms
plot_xmax = 8000
T_K = 8000.0            # temperature in Kelvin (to determine level populations)
sigma_v = 2500.0         # Gaussian width in km/s
Fe3overFe2 = 2.3        # number ratio of these ions

print_lines = False     # output the details of each transition in the plot range to the standard output

plot_resolution = int((plot_xmax-plot_xmin)/1000) # resolution of the plot in Angstroms

gaussian_window = 4     # Gaussian line profile are zero beyond __ sigmas from the centre

# also calculate wavelengths outside the plot range to include lines whose edges pass through the plot range
plot_xmin_wide = plot_xmin * (1 - gaussian_window * sigma_v / c)
plot_xmax_wide = plot_xmax * (1 + gaussian_window * sigma_v / c)

ion = namedtuple('ion', 'ion_stage number_fraction')

fe_ions = [
           ion(1, 0.3),
           ion(2, 1/(1 + Fe3overFe2)),
           ion(3, Fe3overFe2/(1 + Fe3overFe2)),
           ion(4, 0)
          ]

o_ions = [ion(1, 0.5),
          ion(2, 0.5)]

co_ions = [ion(2, 1.0)]

elementslist = [(8,o_ions),(26,fe_ions),(27,co_ions)]

elsymbols = ('','H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P',
             'S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu',
             'Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc',
             'Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La',
             'Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu',
             'Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At',
             'Rn','Fr','Ra','Ac')


roman_numerals = ('','I','II','III','IV','V','VI','VII','VIII','IX','X','XI','XII',
                  'XIII','XIV','XV','XVI','XVII','XVIII','XIX','XX')

def main():
    for (atomic_number,ions) in elementslist:
        elsymbol = elsymbols[atomic_number]
        transition_file = 'transitions_{}.txt'.format(elsymbol)
        print('Loading {:}...'.format(transition_file))
        transitions = load_transitions(transition_file)

        #filter the line list
        transitions = transitions[
            (transitions[:]['lambda_angstroms'] >= plot_xmin_wide) &
            (transitions[:]['lambda_angstroms'] <= plot_xmax_wide) &
            (transitions[:]['forbidden'] == 1) #&
            #(transitions[:]['upper_has_permitted'] == 0)
        ]

        print('{:d} matching lines in plot range'.format(len(transitions)))

        print('Generating spectra...')
        xvalues, yvalues = generate_spectra(transitions,atomic_number,ions)

        print('Plotting...')
        make_plot(xvalues,yvalues,elsymbol,ions)

def load_transitions(transition_file):
    if os.path.isfile(transition_file + '.tmp'):
        #read the sorted binary file (fast)
        transitions = pd.read_pickle(transition_file + '.tmp')
    else:
        #read the text file (slower)
        transitions = pd.read_csv(transition_file, delim_whitespace=True)
        transitions.sort_values(by='lambda_angstroms', inplace=True)

        #save the dataframe in binary format for next time
        # transitions.to_pickle(transition_file + '.tmp')

    return transitions

def generate_spectra(transitions,atomic_number,ions):
    xvalues = np.arange(plot_xmin_wide,plot_xmax_wide,step=plot_resolution)
    yvalues = np.zeros((len(ions), len(xvalues)))

    #iterate over lines
    for index, line in transitions.iterrows():
        flux_factor = f_flux_factor(line)

        ion_index = -1
        for tmpion_index in range(len(ions)):
            if atomic_number == line['Z'] and ions[tmpion_index].ion_stage == line['ion_stage']:
                ion_index = tmpion_index
                break

        if ion_index != -1:
            if print_lines:
                print_line_details(line)

            #contribute the Gaussian line profile to the discrete flux bins

            centre_index = int(round((line['lambda_angstroms'] - plot_xmin_wide) / plot_resolution))
            sigma_angstroms = line['lambda_angstroms'] * sigma_v / c
            sigma_gridpoints = int(math.ceil(sigma_angstroms / plot_resolution))
            window_left_index = max(int(centre_index - gaussian_window * sigma_gridpoints), 0)
            window_right_index = min(int(centre_index + gaussian_window * sigma_gridpoints), len(xvalues))

            for x in range(window_left_index,window_right_index):
                yvalues[ion_index][x] += flux_factor * math.exp(-((x - centre_index)*plot_resolution/sigma_angstroms) ** 2) / sigma_angstroms

    return xvalues, yvalues

def f_flux_factor(line):
    return line['A'] * line['upper_statweight'] * math.exp(-line['upper_energy_Ev']/K_B/T_K)

def print_line_details(line):
    print('lambda {:7.1f}, Flux {:8.2E}, {:2s} {:3s} {:}, {:}'.format(
        line['lambda_angstroms'], f_flux_factor(line), elsymbols[line['Z']],
        roman_numerals[line['ion_stage']], ['permitted','forbidden'][line['forbidden']],
        ['upper state has no permitted lines','upper state has permitted lines'][line['upper_has_permitted']]))
    return

def make_plot(xvalues,yvalues,elsymbol,ions):
    import matplotlib
    matplotlib.use('PDF')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(len(ions)+1, 1, sharex=True, figsize=(6,6), tight_layout={"pad":0.2, "w_pad":0.0, "h_pad":0.0})

    yvalues_combined = np.zeros(len(xvalues))
    for ion_index in range(len(ions)+1):
        if ion_index < len(ions): #an ion subplot
            if max(yvalues[ion_index]) > 0.0:
                yvalues_normalised = yvalues[ion_index] / max(yvalues[ion_index])
                yvalues_combined += yvalues_normalised * ions[ion_index].number_fraction
            else:
                yvalues_normalised = yvalues[ion_index]
            ax[ion_index].plot(xvalues, yvalues_normalised, linewidth=1.5,
                               label='{0} {1}'.format(elsymbol, roman_numerals[ions[ion_index].ion_stage]))

        else: # the subplot showing combined spectrum of multiple ions and observational data
            dir = os.path.dirname(__file__)
            obsspectra = [
                #('dop_dered_SN2013aa_20140208_fc_final.txt','SN2013aa +360d (Maguire)','0.3'),
                #('2010lp_20110928_fors2.txt','SN2010lp +264d (Taubenberger et al. 2013)','0.1'),
                ('2003du_20031213_3219_8822_00.txt', 'SN2003du +221.3d (Stanishev et al. 2007)','0.0'),
                         ]

            for (filename, serieslabel, linecolor) in obsspectra:
              obsfile = os.path.join(dir, 'spectra',filename)
              obsdata = pd.read_csv(obsfile, delim_whitespace=True, header=None, names=['lambda_angstroms','flux'])
              obsdata = obsdata[(obsdata[:]['lambda_angstroms'] > plot_xmin) & (obsdata[:]['lambda_angstroms'] < plot_xmax)]
              obsyvalues = obsdata[:]['flux'] * max(yvalues_combined) / max(obsdata[:]['flux'])
              ax[-1].plot(obsdata[:]['lambda_angstroms'], obsyvalues, lw=1, color='black', label=serieslabel, zorder=-1)

            combined_label = ' + '.join(['({0:.1f} * {1} {2})'.format(ion.number_fraction, elsymbol, roman_numerals[ion.ion_stage]) for ion in ions])
            ax[-1].plot(xvalues, yvalues_combined, lw=1.5, label=combined_label)
            ax[-1].set_xlabel(r'Wavelength ($\AA$)')

        ax[ion_index].set_xlim(xmin=plot_xmin, xmax=plot_xmax)
        ax[ion_index].legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': 10})
        ax[ion_index].set_ylabel(r'$\propto$ F$_\lambda$')

    #ax.set_ylim(ymin=-0.05,ymax=1.1)

    fig.savefig('transitions_{}.pdf'.format(elsymbol),format='pdf')
    plt.close()

main()
