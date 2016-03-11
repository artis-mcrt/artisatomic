#!/usr/bin/env python3
import collections
import os
import sys
import numpy as np
import math
import re

elsymbols = ['n']+[line.split()[1] for line in open('atomic_symbols.dat')]

romannumerals = ('','I','II','III','IV','V','VI','VII','VIII','IX','X','XI','XII',
                 'XIII','XIV','XV','XVI','XVII','XVIII','XIX','XX')

NPHIXSPOINTS = 100
NPHIXSNUINCREMENT = 0.1

def isfloat(value):
  try:
    float(value.replace('D','E'))
    return True
  except ValueError:
    return False

HOVERKB = 4.799243681748932e-11
RYD = 2.1798741e-11 # Rydberg to ergs
H = 6.6260755e-27 # Planck constant in erg seconds

def smoothphixslist(listin, temperature): #list of points (energy, phixs cross section)
    listout = []
    xgrid = np.arange(1.0,1.0+NPHIXSNUINCREMENT*(NPHIXSPOINTS+1),NPHIXSNUINCREMENT)
#    xgrid = list(filter(lambda x:x < (listin[-1][0]/listin[0][0]),xgrid))
    for i in range(len(xgrid)-1):
        enlow = xgrid[i]*listin[0][0]
        enhigh = xgrid[i+1]*listin[0][0]
        
        listenergyryd = np.linspace(enlow,enhigh,num=NPHIXSNUINCREMENT*5000,endpoint=False)
        dnu = (listenergyryd[1] - listenergyryd[0]) * RYD / H

        listsigma_bfMb = np.interp(listenergyryd,*zip(*listin),right=-1)

        integralnosigma = 0.0
        integralwithsigma = 0.0
        nu0 = listenergyryd[0] * RYD / H
        previntegrand = nu0 ** 2
        for j in range(1,len(listenergyryd)):
            energyryd = listenergyryd[j]
            sigma_bfMb = listsigma_bfMb[j]
            if sigma_bfMb < 0:
                sigma_bf = (listin[-1][0]/energyryd) ** 3 * listin[-1][1] #extrapolation
            else:
                sigma_bf = sigma_bfMb
            nu = energyryd * RYD / H
            
            integrandnosigma = (nu ** 2) * math.exp(-HOVERKB*(nu-nu0)/temperature)
            integralcontribution = (integrandnosigma + previntegrand) / 2.0
            integralnosigma += integralcontribution
            integralwithsigma += integralcontribution * sigma_bf
            previntegrand = integrandnosigma
            
        if integralnosigma > 0:
            listout.append( (integralwithsigma/integralnosigma) )
        else:
            listout.append( (0.0) )
            print('probable underflow')

    return listout[:NPHIXSPOINTS]

def weightedavgenergyinev(listenergylevelsthision,ids):
    sum = 0.0
    gsum = 0.0
    for id in ids:
        statisticalweight = float(listenergylevelsthision[id].g)
        sum += statisticalweight * hcinevcm * float(listenergylevelsthision[id].energyabovegsinpercm)
        gsum += statisticalweight
    return sum / gsum

def weightedavthresholdinev(listenergylevelsthision,ids):
    sum = 0.0
    gsum = 0.0
    for id in ids:
        statisticalweight = float(listenergylevelsthision[id].g)
        sum += statisticalweight * hcinevangstrom / float(listenergylevelsthision[id].lambdaangstrom)
        gsum += statisticalweight
    return sum / gsum


def shortenconfig(strin):
    strout = re.sub('[\(\[].*?[\)\]]', '', strin)
    strout = strout.replace(' ','')
    for p in range(len(strout)):
        if strout[p:p+2] == '0s':
            if strout[p-1].isdigit():
                strout = strout[:p-1] + str(int(strout[p-1])+1) + strout[p+2:]
                break
            else:
                break
    return strout

alphabets = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ '
reversedalphabets = 'zyxwvutsrqponmlkjihgfedcbaZYXWVUTSRQPONMLKJIHGFEDCBA '
lchars = 'SPDFGHIKLMNOPQRSTUVWXYZ'

def naharstatefromconfig(truncatedlevelname):
    for i in reversed(range(len(truncatedlevelname))):
        if truncatedlevelname[i] in lchars:
            lposition = i
            l = lchars.index(truncatedlevelname[i])
            break

    twosplusone = int(truncatedlevelname[lposition-1]) #could this be two digits long?
    if lposition + 1 > len(truncatedlevelname) - 1:
        parity = 0
    elif truncatedlevelname[lposition+1] == 'o':
        parity = 1
    elif truncatedlevelname[lposition+1] == 'e':
        parity = 0
    elif truncatedlevelname[lposition+2] == 'o':
        parity = 1
    elif truncatedlevelname[lposition+2] == 'e':
        parity = 0
    else:
        print("Can't interpret term for",truncatedlevelname)
        twosplusone = -1
        l = -1
        parity = -1
#        sys.exit()
    return (twosplusone, l, parity)

if __name__ == "__main__":
    hcinevcm = 1.23984193e-4
    hcinevangstrom = 1.23984193e4
    hinevsecond = 4.135667662e-15

    listions = [
            (26, 1),
            (26, 2),
            (26, 3),
            (26, 4),
            (26, 5)
        ]
    
    naharcorestaterow = collections.namedtuple('naharcorestate','id configuration term energyrydberg upperlevelid')
    naharcorestates = [['IGNORE'] for x in listions] #list of named tuples (naharcorestaterow)
    naharconfigs = [{} for x in listions] #keys are (2s+1, l, parity, indexinsymmetry), values are strings of electron configuration
    
    energylevelrow = collections.namedtuple('energylevel','indexinsymmetry TC corestateid elecn elecl energyrydberg twosplusone l parity transitioncount')
    listenergylevels = [['IGNORE'] for x in listions] #list of named tuples (first element is IGNORE to discard 0th index)
    
    transitionrow = collections.namedtuple('transition','namefrom nameto f A lambdaangstrom i j id')
    listtransitions = [['IGNORE'] for x in listions] #list of named tuples (transitionrow)
    transitioncountofenergylevelname = [{} for x in listions]
    
    ionizationpotentialrydberg = [0.0 for x in listions]
    groundstateenergyev = [0.0 for x in listions]
    photoionizations = [{} for x in listions] #keys are (2s+1, l, parity, indexinsymmetry), values are lists of (energy in Rydberg, cross section in Mb) tuples

    for i in range(len(listions)):
        (atomicnumber, ionizationstage) = listions[i]

        print('==============> {0} {1}:'.format(elsymbols[atomicnumber],romannumerals[ionizationstage]))
        
        #read Nahar energy level file
        pathnaharen = 'atomic-data-nahar/{0}{1:d}.en.ls.txt'.format(elsymbols[atomicnumber].lower(),ionizationstage)
        print('Reading ' + pathnaharen)
        
        with open(pathnaharen, 'r') as fenlist:
            while True:
                line = fenlist.readline()
                if not line:
                    print('End of file before data section')
                    sys.exit()
                if line.startswith(' i) Table of target/core states in the wavefunction expansion'):
                    break

            while True:
                line = fenlist.readline()
                if not line:
                    print('End of file before end of core states table')
                    sys.exit()
                if line.startswith(' no of target/core states in WF'):
                    numberofcorestates = int(line.split('=')[1])
                    break
            fenlist.readline() #blank line
            fenlist.readline() #' target states and energies:'
            fenlist.readline() #blank line
            for n in range(numberofcorestates):
                row = fenlist.readline().split()
                naharcorestates[i].append(naharcorestaterow._make( (int(row[0]),row[1],row[2],float(row[3]),-1) ))
                if int(naharcorestates[i][-1].id) != len(naharcorestates[i])-1:
                        print('Nahar levels mismatch: id {0:d} found at entry number {1:d}'.format(
                            len(naharcorestates[i])-1,int(naharcorestates[i][-1].id)))
                        sys.exit()
            
            while True:
                line = fenlist.readline()
                if not line:
                    print('End of file before data section ii)')
                    sys.exit()
                if line.startswith('ii) Table of bound (negative) state energies (with spectroscopic notation)'):
                    break
        
            foundtable = False
            while True:
                line = fenlist.readline()
                if not line:
                    print('End of file before end of state table')
                    sys.exit()
                if line.startswith(' Ion ground state'):
                    ionizationpotentialrydberg[i] = -float(line.split('=')[2])
                    print("IP={0:.4f} Ry".format(ionizationpotentialrydberg[i]))
                if line.startswith(' ') and len(line) > 36 and isfloat(line[29:29+8]):
                    foundtable = True
                    state = line[1:22]
                    energy = float(line[29:29+8])
                    twosplusone = int(state[18])
                    l = lchars.index(state[19])
                    
                    if state[20] == 'o':
                        parity = 1
                        indexinsymmetry = reversedalphabets.index(state[17])+1
                    else:
                        parity = 0
                        indexinsymmetry = alphabets.index(state[17])+1
                    
                    #print(state,energy,twosplusone,l,parity,indexinsymmetry)
                    naharconfigs[i][(twosplusone,l,parity,indexinsymmetry)] = state
                else:
                    if foundtable: break
                    
            while True:
                line = fenlist.readline()
                if not line:
                    print('End of file before table iii header')
                    sys.exit()
                if line.startswith('iii) Table of complete set (both negative and positive) of energies'):
                    break
            line = fenlist.readline() #line of ----------------------
            while True:
                line = fenlist.readline()
                if not line:
                    print('End of file before table iii starts')
                    sys.exit()
                if line.startswith('------------------------------------------------------'):
                    break

            row = fenlist.readline().split() #line of atomic number and electron number, unless...
            while len(row)==0: #some extra blank lines might exist
                row = fenlist.readline().split()
            
            if atomicnumber != int(row[0]) or ionizationstage != int(row[0]) - int(row[1]):
                    print('Wrong atomic number or ionization stage in Nahar energy file',atomicnumber,int(row[0]),ionizationstage,int(row[0]) - int(row[1]))
                    sys.exit()
            
            while True:
                line = fenlist.readline()
                if not line:
                    print('End of file before table iii finished')
                    sys.exit()
                row = line.split()
                if '-'.join(row) == '0-0-0-0':
                    break
                twosplusone = int(row[0])
                l = int(row[1])
                parity = int(row[2])
                numberofstatesinsymmetry = int(row[3])
                
                line = fenlist.readline() #core state number and energy for the whole LSP group. Not sure what to do with this.
                
                for s in range(numberofstatesinsymmetry):
                    row = fenlist.readline().split()
                    listenergylevels[i].append(energylevelrow._make(row+[twosplusone,l,parity,0]))
                    if float(listenergylevels[i][-1].energyrydberg) >= 0.0:
                        listenergylevels[i].pop()

            listenergylevels[i] = ['IGNORE'] + sorted(listenergylevels[i][1:],key=lambda x:float(x.energyrydberg))
#                    print(listenergylevels[i][-1])
            
        #end reading Nahar energy file
        
        #read Nahar linelist
        """
        pathnaharf = 'atomic-data-nahar/{0}{1:d}.f.ls.txt'.format(elsymbols[atomicnumber].lower(),ionizationstage)
        print('Reading ' + pathnaharen)
        with open(pathnaharf,'r') as fnaharf:
            for line in fnaharf:
                row = line.split()
                
                # check for right number of columns and are all numbers except first column
                if len(row) == len(energylevelrow._fields) and all(map(isfloat, row[1:])):
                    listenergylevels[i].append(energylevelrow._make(row))

                    transitioncountofenergylevelname[i][listenergylevels[i][-1].name] = 0
                    levelid = int(listenergylevels[i][-1].id.lstrip('-'))
                    idofenergylevelname[i][listenergylevels[i][-1].name] = levelid
#                            if '_' not in truncatedlevelname:
#                                print('No term? ',truncatedlevelname)
                    (twosplusone,l,parity) = naharstatefromconfig(truncatedlevelname)
                    if twosplusone != -1: #value means that the state could not be interpreted
                        #match up nahar and hiller states
                        if (twosplusone,l,parity) in truncatedlevelnamesofnaharstate[i].keys():
                            if truncatedlevelname not in truncatedlevelnamesofnaharstate[i][(twosplusone,l,parity)]:
                                truncatedlevelnamesofnaharstate[i][(twosplusone,l,parity)].append(truncatedlevelname)
                            #print(truncatedlevelname,naharstatefromconfig(truncatedlevelname),)
                        else:
                            truncatedlevelnamesofnaharstate[i][(twosplusone,l,parity)] = [truncatedlevelname,]
                        
                        if float(listenergylevels[i][-1].energyabovegsinpercm) < 1.0: #if this is the ground state
                            ionizationenergyev[i] = hcinevangstrom / float(listenergylevels[i][-1].lambdaangstrom)
#                                ionizationenergyev[i] = float(listenergylevels[i][-1].thresholdenergyev)

                if line.startswith('                        Oscillator strengths'):
                    break
        """
        #end reading Nahar linelist

        #read Nahar photoionization cross sections
        pathphot = 'nahar_radiativeatomicdata/{0}{1:d}.px.txt'.format(elsymbols[atomicnumber].lower(),ionizationstage)
        print('Reading ' + pathphot)
    
        with open(pathphot, 'r') as fenlist:
            while True:
                line = fenlist.readline()
                if not line:
                    print('End of file before data section')
                    sys.exit()
                if line.startswith('----------------------------------------------'):
                    break
            
            line = fenlist.readline()
            row = line.split()
            if atomicnumber != int(row[0]) or ionizationstage != int(row[0]) - int(row[1]):
                    print('Wrong atomic number or ionization stage in Nahar file',atomicnumber,int(row[0]),ionizationstage,int(row[0]) - int(row[1]))
                    sys.exit()

            numbernaharstates = 0
            while True:
                line = fenlist.readline()
                row = line.split()
                if not line or sum(map(float,row))==0:
                    break
                numbernaharstates += 1
                twosplusone = int(row[0])
                l = int(row[1])
                parity = int(row[2])
                indexinsymmetry = int(row[3])
                line = fenlist.readline()
                numberofpoints = int(line.split()[1])
                naharbindingenergyrydberg = float(fenlist.readline().split()[0])   #line containing binding energy

                lowerlevelid = (twosplusone,l,parity,indexinsymmetry)
                photoionizations[i][lowerlevelid] = []

                for p in range(numberofpoints):
                    row = fenlist.readline().split()
                    photoionizations[i][lowerlevelid].append( (float(row[0]),float(row[1])) )
                    
                    if (len(photoionizations[i][lowerlevelid]) > 1 and
                        photoionizations[i][lowerlevelid][-1][0] <= photoionizations[i][lowerlevelid][-2][0]):
                        #some (energy) x values are only repeated because they're not specified with high enough precision
                        #just removing these is not worth reporting
                        #if abs(photoionizations[i][lowerlevelid][-1][0] - photoionizations[i][lowerlevelid][-2][0]) > 1e-3:
                            #print('ERROR: photoionization table first column not monotonically increasing',truncatedlowerlevelname,twosplusone,l,parity,indexinsymmetry)
                            #print(row)
                        photoionizations[i][lowerlevelid].pop()
        #end reading Nahar photoionization cross sections

    with open('adata.txt', 'w') as fatommodels, open('transitiondata.txt', 'w') as flinelist, open('phixsdata2.txt', 'w') as fphixs:
        print('\nStarting output stage:'.format(atomicnumber,ionizationstage))
        fphixs.write('{0:d}\n'.format(NPHIXSPOINTS))
        fphixs.write('{0:14.7e}\n'.format(NPHIXSNUINCREMENT))
        for i in range(len(listions)):
            (atomicnumber, ionizationstage) = listions[i]
            
            print('==============> {0} {1}:'.format(elsymbols[atomicnumber],romannumerals[ionizationstage]))

            # write output files for artis
            print("writing to 'adata.txt'")
            fatommodels.write('{0:12d}{1:12d}{2:12d}{3:15.7f}\n'.format(atomicnumber,ionizationstage,len(listenergylevels[i])-1,ionizationpotentialrydberg[i]*13.605698066))

            for energylevelid in range(1,len(listenergylevels[i])):
                energylevel = listenergylevels[i][energylevelid]
                fatommodels.write('{0:7d}{1:25.16f}{2:25.16f}{3:7d}\n'.format(energylevelid,(ionizationpotentialrydberg[i]+float(energylevel.energyrydberg))*13.605698066,energylevel.twosplusone*(2*energylevel.l+1),energylevel.transitioncount))
            fatommodels.write('\n')
            
            print("writing to 'transitiondata.txt'")
            flinelist.write('{0:7d}{1:7d}{2:12d}\n'.format(atomicnumber,ionizationstage,len(listtransitions[i])-1))
            for transition in listtransitions[i][1:]:
                flinelist.write('{0:12d}{1:7d}{2:12d}{3:25.16E} -1\n'.format(int(transition.id),idofenergylevelname[i][transition.namefrom],idofenergylevelname[i][transition.nameto.lstrip('-')],float(transition.A)))
            flinelist.write('\n')
            
            print("writing to 'phixsdata2.txt'")
            if i < len(listions)-1:
                for corestateindex in range(1,len(naharcorestates[i])):
                    thiscorestate = naharcorestates[i][corestateindex]
                    candidatematchesgoodtermgoodconfig = [] #list of tuples (energylevelindex, energyratio)
                    candidatematchesgoodtermwildcardconfig = [] #list of tuples (energylevelindex, energyratio)
                    candidatematchesgoodterm = [] #list of tuples (energylevelindex, energyratio)
                    candidatematchesother = [] #list of tuples (energylevelindex, energyratio)
                    for upperlevelindex in range(1,len(listenergylevels[i+1])): #need to put these in energy order somehow
                        upperenergylevel = listenergylevels[i+1][upperlevelindex]
                        twosplusone = int(upperenergylevel.twosplusone)
                        l = int(upperenergylevel.l)
                        parity = int(upperenergylevel.parity)
                        indexinsymmetry = int(upperenergylevel.indexinsymmetry)
                        if (twosplusone,l,parity,indexinsymmetry) in naharconfigs[i+1]:
                            upperlevelconfig = naharconfigs[i+1][(twosplusone,l,parity,indexinsymmetry)][:16]
                            #print(config,shortenconfig(config))

                            if thiscorestate.term[0] == '{0:d}'.format(twosplusone) and thiscorestate.term[1] == lchars[l] and \
                                ((parity == 0 and len(thiscorestate.term) < 3) or (parity == 1 and len(thiscorestate.term) >= 3 and thiscorestate.term[2] == 'o')):
                                #print(config,'|',lowerlevelcorestate.configuration)
                                upperlevelenergyabovegs = (ionizationpotentialrydberg[i+1]+float(upperenergylevel.energyrydberg))
                                corestateenergyabovegs = float(thiscorestate.energyrydberg)
                                energydeviation = abs(corestateenergyabovegs - upperlevelenergyabovegs)
                                if shortenconfig(upperlevelconfig) == thiscorestate.configuration:
                                    candidatematchesgoodtermgoodconfig.append( (upperlevelindex, energydeviation) )
                                elif "0s" in upperlevelconfig:
                                    candidatematchesgoodtermwildcardconfig.append( (upperlevelindex, energydeviation) )
                                else:
                                    candidatematchesgoodterm.append( (upperlevelindex, energydeviation) )
                            else:
                                candidatematchesother.append( (upperlevelindex, energydeviation) )
                        else:
                            candidatematchesother.append( (upperlevelindex, energydeviation) )
                
#                        else:
#                            print('no config for ',(twosplusone,l,parity,indexinsymmetry))
                    sortedmatches = sorted(candidatematchesgoodtermgoodconfig,key=lambda x:x[1]) + sorted(candidatematchesgoodtermwildcardconfig,key=lambda x:x[1]) + sorted(candidatematchesgoodterm,key=lambda x:x[1]) + sorted(candidatematchesother,key=lambda x:x[1])
                    for candidate in sortedmatches:
                        if candidate[0] not in [cs.upperlevelid for cs in naharcorestates[i][1:]]:
                            naharcorestates[i][corestateindex] = naharcorestates[i][corestateindex]._replace(upperlevelid=candidate[0])
                            break
                
                #debugging info only, not needed
                #output a list of energy levels and their corresponding core states
                """
                if ionizationstage == 2:
                    for upperlevelindex in range(1,len(listenergylevels[i+1])):
                        upperenergylevel = listenergylevels[i+1][upperlevelindex]
                        upperlevelenergyabovegs = (ionizationpotentialrydberg[i+1]+float(upperenergylevel.energyrydberg))
                        twosplusone = int(upperenergylevel.twosplusone)
                        l = int(upperenergylevel.l)
                        parity = upperenergylevel.parity
                        indexinsymmetry = int(upperenergylevel.indexinsymmetry)
                        corestateindex = -1
                        for csid in range(1,len(naharcorestates[i])):
                            if int(naharcorestates[i][csid].upperlevelid) == upperlevelindex:
                                corestateindex = csid
                                break
                        if corestateindex != -1:
                            thiscorestate = naharcorestates[i][corestateindex]
                            corestateenergyabovegs = float(thiscorestate.energyrydberg)
                            
#                            print('{0}   {1:.5f} {2:12.6E}   core state {3}, with E={4:12.6E}'.format(naharconfigs[i+1][(twosplusone,l,parity,indexinsymmetry)],float(upperenergylevel.energyrydberg),upperlevelenergyabovegs,thiscorestate.id,thiscorestate.energyrydberg))
                            print('{0}  {1}'.format(naharconfigs[i+1][(twosplusone,l,parity,indexinsymmetry)],thiscorestate.id))
                        else:
#                            print('{0}   {1:.5f} {2:12.6E}   core state'.format(naharconfigs[i+1][(twosplusone,l,parity,indexinsymmetry)],float(upperenergylevel.energyrydberg),upperlevelenergyabovegs))
                            print('{0}  '.format(naharconfigs[i+1][(twosplusone,l,parity,indexinsymmetry)]))
                """

                if ionizationstage == 2:
                    for levelindex in range(1,len(listenergylevels[i])):
                        energylevel = listenergylevels[i][levelindex]
                        twosplusone = int(energylevel.twosplusone)
                        l = int(energylevel.l)
                        parity = energylevel.parity
                        indexinsymmetry = int(energylevel.indexinsymmetry)
                        print('{0}@{1}'.format(levelindex-1,naharconfigs[i][(twosplusone,l,parity,indexinsymmetry)]))
                
                for lowerlevelidnum in range(1,len(listenergylevels[i])):
                    lowerlevel = listenergylevels[i][lowerlevelidnum]
                    if 0 < int(lowerlevel.corestateid) < len(naharcorestates[i])-1:
                        lowerlevelcorestate = naharcorestates[i][int(lowerlevel.corestateid)]
                    else:
                        lowerlevelcorestate = naharcorestates[i][1]
                        print('No core state for L,S,P,index=',(lowerlevel.twosplusone,lowerlevel.l,lowerlevel.parity,lowerlevel.indexinsymmetry),int(lowerlevel.corestateid))
                    #(twosplusone, l, parity) = naharstatefromconfig(lowerlevelcorestate.term)
                    lowerlevelid = (lowerlevel.twosplusone,lowerlevel.l,lowerlevel.parity,int(lowerlevel.indexinsymmetry))
    #                lowerlevelidstring = '{0}{1}{2}{3}'.format(alphabets[int(lowerlevel.indexinsymmetry)-1],lowerlevel.twosplusone,lchars[lowerlevel.l],('e','o')[lowerlevel.parity],)
    
                    upperlevelid = lowerlevelcorestate.upperlevelid
                    if upperlevelid == -1:
                        upperlevelid = 1
                    
                    thresholdenergyrydberg = -float(lowerlevel.energyrydberg) #energy up to ionisation
                    thresholdenergyrydberg += float(lowerlevelcorestate.energyrydberg) #plus upper level energy
                    
                    if lowerlevelid in photoionizations[i].keys():
                        smoothedphixs = smoothphixslist(photoionizations[i][lowerlevelid], 1000) #temperature here is important
                    else:
                        smoothedphixs = [0.0] * NPHIXSPOINTS
                    
                    fphixs.write('{0:12d}{1:12d}{2:8d}{3:12d}{4:8d}\n'.format(atomicnumber,ionizationstage+1,upperlevelid,ionizationstage,lowerlevelidnum))
                    
                    for crosssection in smoothedphixs:
                        fphixs.write('{0:16.8E}\n'.format(crosssection))
