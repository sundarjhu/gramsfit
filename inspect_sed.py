from astropy.table import Table, join
import numpy as np
import os
import matplotlib.pyplot as plt

def setPlotParams():
    plt.figure(figsize = (8, 8))
    params = {'legend.fontsize': 'x-large',
              'axes.labelsize':20,
              'axes.titlesize':20,
              'xtick.labelsize':20,
              'ytick.labelsize':20,
              'text.usetex': True,
              'text.latex.preamble': r'\usepackage{bm}'}
    plt.rcParams.update(params)
    return plt

def plot_sed(t, wavefile = 'pivot_wavelengths.csv'):
    """
    Given the table of all possible AllWISE+2MASS+GaiaDR2 matches within 60" to each IRAS+AKARI source, 
        inspect the SEDs to assign the AllWISE "best match" to each IRAS+AKARI source.
    By default, the AllWISE source with W3 and W4 flux closest to the IRAS12 and IRAS25 fluxes is considered.
        This choice can then be modified by the user based on visual inspection of the SEDs.
    INPUT:
        t -- Astropy table containing the fluxes and designations.
        wavefile -- link to VOTable containing the pivot wavelengths.
    OUTPUT:
        inspect -- Astropy table containing the AllWISE designation of the "best match", and any user-specified comments on
        the match.
    """
    ndata = len(t)
    #A measure for the discrepancy between (IRAS12, IRAS25) and (WISEW3, WISEW4)
    dflux1 = np.array(t['psc12'] - t['AllWISE_W3_flux'])**2; dflux2 = np.array(t['psc25'] - t['AllWISE_W4_flux'])**2
    IRAS_WISE_flux_discrep = np.nansum([dflux1, dflux2], axis = 0)/(np.isfinite(dflux2)*1 + np.isfinite(dflux1)*1) #kind of a chisq per valid point

    l = Table.read(wavefile, format = 'csv')
    #Fluxes for the matches in Abrahamyan+ 2015 (these stay the same for a given IRAS PSC source)
    lAb2015 = np.array(l['pivot_wavelength'][4:14]) #The corresponding wavelengths, not counting the FSC points for NESS
    ks = np.argsort(lAb2015) #Sorted so connecting lines in plot are unidirectional
    lAb2015 = lAb2015[ks]
    fAb2015 = np.array([t['psc12'], t['psc25'], t['psc60'], t['psc100'], t['S09'], t['S18'], t['S65'], t['S90'], t['S140'], t['S160']])
    e_fAb2015 = np.array([t['e_psc12'], t['e_psc25'], t['e_psc60'], t['e_psc100'], \
                          t['e_S09'], t['e_S18'], t['e_S65'], t['e_S90'], t['e_S140'], t['e_S160']])
    #Sort the fluxes
    fAb2015 = fAb2015[ks, :]; e_fAb2015 = e_fAb2015[ks, :]

    #Fluxes for the matches performed in this work 
    lmat = np.array(l['pivot_wavelength'][14:]) #The corresponding wavelengths, not counting the FSC points for NESS
    ks = np.argsort(lmat) #Sorted so connecting lines in plot are unidirectional
    lmat = lmat[ks]
    fmat = np.array([t['AllWISE_W1_flux'], t['AllWISE_W2_flux'], t['AllWISE_W3_flux'], t['AllWISE_W4_flux'], \
                     t['TMASS_J_flux'], t['TMASS_H_flux'], t['TMASS_K_flux'], \
                     t['GaiaDR2_BP_mean_flux'], t['GaiaDR2_G_mean_flux'], t['GaiaDR2_RP_mean_flux']])
    e_fmat = np.array([t['AllWISE_W1_e_flux'], t['AllWISE_W2_e_flux'], t['AllWISE_W3_e_flux'], t['AllWISE_W4_e_flux'], \
                       t['TMASS_J_e_flux'], t['TMASS_H_e_flux'], t['TMASS_K_e_flux'], \
                       t['GaiaDR2_BP_mean_e_flux'], t['GaiaDR2_G_mean_e_flux'], t['GaiaDR2_RP_mean_e_flux']])
    #Sort the fluxes
    fmat = fmat[ks, :]; e_fmat = e_fmat[ks, :]

    _, u, ui = np.unique(t['IRAS_CNTR'], return_index = True, return_inverse = True)

    inspect = Table([t[u]['IRAS_CNTR'], t[u]['IRAS-PSC'], t[u]['AllWISE']], names = ('IRAS_CNTR', 'IRAS-PSC', 'AllWISE'))
    inspect['comments'] = ' '*120

    plt = setPlotParams()
    #Check if matplotlib can access the user's TeX distribution
    try:
        _ = plt.xlabel(r'$\lambda$ ($\mu$' + 'm)'); _ = plt.ylabel(r'$F_\nu$ (' + 'Jy)')
    except:
        _ = plt.xlabel('Wavelength (micron)'); _ = plt.ylabel('Flux density (Jy)')
    #
    plt.ion()
    #while loop to allow returning to the previous SED in case the user made a mistake
    k = 0
    while k < len(u):
        if k != len(u) - 1:
            up = u[k+1]
        else:
            up = ndata
        #Sort in ascending order of flux disagreement between (IRAS12, IRAS25) and (WISEW3, WISEW4)
        ks = np.argsort(IRAS_WISE_flux_discrep[u[k]:up])
        kk = np.arange(u[k], up)[ks]
        #First plot the IRAS + AKARI fluxes
        _ = plt.errorbar(lAb2015, fAb2015[:, u[k]], yerr = e_fAb2015[:, u[k]], fmt = 'ko', linestyle = '-')
        #Loop over the matches and plot the AllWISE/2MASS/GaiaDR2 fluxes
        for j in range(len(kk)):
            #_ = plt.errorbar(lmat, fmat[:, j], yerr = e_fmat[:, j], fmt = 'o', linestyle = '-', label = t[j]['AllWISE'])
            _ = plt.errorbar(lmat, fmat[:, kk[j]], yerr = e_fmat[:, kk[j]], fmt = 'o', linestyle = '-', label = str(j).strip())
        _ = plt.xscale('log'); _ = plt.yscale('log')
        _ = plt.title('IRAS PSC = ' + t[u[k]]['IRAS-PSC'])
        #_ = plt.xlim(0.8 * min(lp[np.argsort(lp)]), 1.2 * max(lp[np.argsort(lp)]))
        _ = plt.xlim(0.8 * min([min(lAb2015), min(lmat)]), 1.2 * max([max(lAb2015), max(lmat)]))
        lolim = np.nanmin([np.nanmin(fAb2015[:, u[k]:up]), np.nanmin(fmat[:, u[k]:up])])
        uplim = np.nanmax([np.nanmax(fAb2015[:, u[k]:up]), np.nanmax(fmat[:, u[k]:up])])
        if uplim/lolim > 1e-10:
            lolim = uplim * 1e-10
        _ = plt.ylim(0.8 * lolim, 1.2 * uplim)
        plt.legend(loc = 'best')
        #plt.show()
        plt.show(block = False)
        truenb = input('True counterpart (default: 0, none: -1, return to previous plot: -2): ')
        if truenb.strip() == '':
            truenb = 0
        else:
            truenb = int(truenb)
        if truenb != -2:
            comment = input('Comments (separate with ; if necessary. NO COMMAS PLEASE!): ')
            if truenb == -1:
                inspect[k]['AllWISE'] = ''
            else:
                inspect[k]['AllWISE'] = t[u[k] + truenb]['AllWISE']
            inspect[k]['comments'] = comment
            k = k + 1
        else:
            k = k - 1
        plt.clf()
    plt.close()

    return inspect

def inspect_sed(outfile = 'inspect_sed.csv'):
    """
    Driver for plot_sed. Allows user to choose a subset of sources to inspect. If output file already exists, allows user to use information
        already present in this file.
    """
    t = Table.read('NESS_IRAS_AKARI_AllWISE_TMASS_GaiaDR2_fluxes.vot', format = 'votable')
    _, u = np.unique(t['IRAS_CNTR'], return_index = True)
    nunique = len(u)

    #Does the user want to subset of the data to inspect?
    print('There are a total of ' + str(nunique) + ' sources to inspect.')
    n = input("Enter the range of unique sources (e.g., 155 - 255) you'd like to inspect [ENTER = 0-" + str(nunique - 1) + "]: ") 
    if n == '':
        k = np.arange(nunique)
    else:
        x = np.array([int(j) for j in n.split('-')])
        k = np.arange(x[0], x[1] + 1)
    #remove the rest from t
    j = Table(); j['IRAS_CNTR'] = t[u[k]]['IRAS_CNTR']
    tt = join(t, j, join_type = 'right', keys = 'IRAS_CNTR')
    t = tt.copy()

    #Have the sources in this list already been inspected?
    if os.path.isfile(outfile):
        inspected = Table.read(outfile, format = 'csv')
        #Ask user if they want to reinspect the IRAS IDs that are already in outfile
        response = input('A file named ' + outfile + ' already exists in this folder. Reinspect the sources already in this file? Y/[N]: ')
        if response.strip().upper() != 'Y':
            #Remove the IRAS IDs that are already in the existing file
            a = set(t[u[k]]['IRAS_CNTR']).difference(set(inspected['IRAS_CNTR']))
            j = Table(); j['IRAS_CNTR'] = list(a)
            tt = join(t, j, join_type = 'right', keys = 'IRAS_CNTR')
            t = tt.copy()

    #Group by IRAS_CNTR, then sort by AllWISE_angDist
    ks = np.lexsort((t['AllWISE_angDist'], t['IRAS_CNTR']))
    t = t[ks].copy()

    inspect = plot_sed(t)

    #Combine new results with existing ones if any
    try:
        vstack([inspected, inspect]).write(outfile, overwrite = True, format = 'csv')
    except:
        inspect.write(outfile, overwrite = True, format = 'csv')
