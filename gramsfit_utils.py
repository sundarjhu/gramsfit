from astropy.table import Table
from astropy.io import fits
import numpy as np
import os, subprocess
import pyphot as pyp
import h5py
from matplotlib import pyplot as plt
import warnings
import gramsfit
from astropy import units


def synthphot(lam, fnu, pypLibrary, filterNames=None,
              filterFile=None):
    """
        Given a grid of wavelengths over which flux densities in F_nu units
        are defined, return the synthetic photometry in the filters specified
        by the filterNames list.

        Args:
        lam (ndarray, Quantity): wavelength grid in micron
            must be 1-d array
        fnu (ndarray, Quantity): flux density grid in F_nu units
            can have shape (len(lam), ) or (n_sources, len(lam)), or
            (len(lam), n_sources)
        pypLibrary (str or pyphot.Library): name of the pyphot library file
            or the pyphot library object
        filterNames (list): list of filter names
        filterFile (str): name of the filter file from which to obtain
            the filterNames if not provided

        Returns:
        lpivot (ndarrray, Quantity): pivot wavelengths of the filters with
            units identical to lam. Has shape len(filterNames).
        fphot (Quantity): synthetic photometry in the filters with
            units identical to fnu. Is a 2-d array with shape (n, m)
            where one of n, m is len(filterNames). The other is the
            number of sources (1 or more). The position of the wavelength
            axis is the same as in fnu;
            e.g., if fnu.shape = (n_sources, len(lam)),
            fphot.shape = (n_sources, len(filterNames)).
    """
    # shape check
    sh_fnu = fnu.shape
    flip_shape = False

    if len(sh_fnu) == 1:
        if len(lam) != sh_fnu:
            raise ValueError('lam and fnu must have the same shape!')
        else:
            fnu_ = fnu.reshape((1, len(fnu)))
            flip_shape = True
    else:
        if len(lam) == sh_fnu[0]:
            fnu_ = fnu.T
        elif len(lam) == sh_fnu[1]:
            fnu_ = fnu
        else:
            raise ValueError('At least one dimension of fnu must have\
                    the same length as lam!')

    if pypLibrary is None:
        raise ValueError('pypLibrary must be specified!')
    if isinstance(pypLibrary, str):
        fL = pyp.get_library(fname=pypLibrary)
    else:  # assume it's a pyphot library object
        fL = pypLibrary

    if filterNames is None:
        if filterFile is None:
            raise ValueError('filterNames or filterFile must be specified!')
        f = Table.read(filterFile, format='csv',
                       names=('column', 'filtername'))
        filterNames = [ff['filtername'].replace('/', '_') for ff in f]

    ezlam = lam.value * pyp.unit[lam.unit.to_string()]
    # # The following ensures that the interpolated grid is high enough resolution
    # #       because remember: the GRAMS grid has crappy resolution!
    # ezlam = np.geomspace(lam.value.min(), lam.value.max(), 1001) * pyp.unit[lam.unit.to_string()]
    filters = fL.load_filters(filterNames, interp=True, lamb=ezlam)
    # need to automatically grab the unit for lpivot
    lpivots = np.array([f.lpivot.magnitude for f in filters])
    flamunits = 'W/m**3'
    flam = fnu_.to(flamunits, equivalencies=units.spectral_density(lam))
    ezflam = flam.value * pyp.unit[flamunits]

    # fphot must have shape (n_sources, n_filters)
    fphot = np.zeros((fnu_.shape[0], len(filters)))

    for i, f, lp in zip(range(len(filters)), filters, lpivots):
        ezfp = f.get_flux(ezlam, ezflam, axis=-1).to(flamunits)
        fp = ezfp.magnitude * flam.unit
        fphot[:, i] = fp.to(fnu_.unit,
                            equivalencies=units.spectral_density(
                                lp * lam.unit)).value

    if flip_shape:
        return lpivots, fphot.T
    else:
        return lpivots, fphot


def setPlotParams():
    plt.figure(figsize = (8, 8))
    params = {'legend.fontsize': 'x-large',
              'axes.labelsize':20,
              'axes.titlesize':20,
              'xtick.labelsize':20,
              'ytick.labelsize':20,
              'text.usetex': True,
              'text.latex.preamble': r'\usepackage{bm}',
              'figure.max_open_warning': 0
              }
    plt.rcParams.update(params)
    plt.rcParams.update({'figure.max_open_warning': 0})
    return plt

def makeFilterSet(filterNames = [], infile = 'filters.csv', libraryFile = 'filters.hd5'):
    """Download filter transmission curves from the Spanish Virtual Observatory.
    The resulting filter library is saved as an hd5 file, to be ingested into pyphot.
    INPUTS:
           1) filterNames, a list of the filter names as on the SVO/VOSA site. If specified, infile
           is ignored.
           2) infile, a two-column CSV file of which the second column must contain
           must contain the names of the SVO/VOSA filter files to download.
           The first column is not currently used, but can contain identifying information
           for each filter that connects it back to the data.
           The filter names can be in the order of occurrence in the data, and may include
           repetitions (as it is quite possible that the data is compiled from a number of
           differing sets of observations).
    OUTPUT: libraryFile, an HDF file containing information about all the filters for which this is requested.

    At the moment, the header in the output files only contains the DetectorType specification if the filter
        is a photon counter (DetectorType value = "1"). We read in the entire file and check for the occurrence
        of this line and set the detector type accordingly.
    """
    if filterNames == []:
        tin = Table.read(infile, format = 'csv', names = ('column', 'filtername'))
        filterNames = list(tin['filtername'])
    url = 'http://svo2.cab.inta-csic.es//theory/fps3/fps.php?ID='
    filters = []
    #Each filter is downloaded into a temporary file via curl.
    #   The temporary file is deleted after all the filters are downloaded.
    for f in filterNames:
    #for t in tin:
        print("Downloading filter " + f)
        _ = subprocess.call(['curl', '-o', 'temp.vot', url + f])
        with open('temp.vot') as g:
            content = g.readlines()
        if any("DetectorType" in c for c in content):
            det_type = 'photon'
        else:
            det_type = 'energy'
        temp = Table.read('temp.vot', format = 'votable')
        g = pyp.Filter(np.array(temp['Wavelength']), np.array(temp['Transmission']), \
                       name = f.replace('/','_'), unit = temp['Wavelength'].unit.name, \
                       dtype = det_type)
        filters.append(g)
    _ = os.remove("temp.vot")  # subprocess.call(['rm', 'temp.vot'])
    #Instantiate an hdf5 object to store filter information
    h = h5py.File(libraryFile, 'w')
    h.create_group('filters')
    h.close()
    h = pyp.HDF_Library(source=libraryFile)
    #Add filters to this object, without repetition.
    _, u = np.unique([f.name for f in filters], return_index=True)
    for f in list(np.array(filters)[u]):
        # h.add_filter(f)
        f.write_to("{0:s}".format(h.source),
                   tablename='/filters/{0}'.format(f.name), append=True)

def editgridheader(header, grid, filters):
    """Modify the header to the original grid, mainly for compatibility with the FITS standard
    The pivot wavelengths are written into the header, and can be retrieved as follows:
    grid = Table.read('grams_o.fits', format = 'fits')
    lpivot = np.array([float(grid.meta['FILT_' + str(i+1)].split(',')[1][:-1]) for i in range(len(grid[0]['Fphot']))])
    """
    keys_orig = [k for k in header.keys()]
    values_orig = [v for v in header.values()]
    comments_orig = [v for v in header.comments]
    h = fits.Header()
    loc = [i for i, h in enumerate(header) if '----' in h]
    #loc = [i for i, h in enumerate(header) if 'DESC' in h]
    #loc2 = [i for i, h in enumerate(header) if 'TTYPE1' in h]
    #loc = [loc[0], loc2[0] - 2]
    for i in range(loc[0]):
        h.append((keys_orig[i], values_orig[i], comments_orig[i]), end = True)
        #h[keys_orig[i]] = values_orig[i]
    #the following lists are first sized by the non-filter elements
    keys = ['DESC', 'GRAMSREF', 'NMODELS', 'PHOTREF', 'DUSTTYPE', 'OPTCREF', 'SIZEDIST', 'WLRANGE', 'DISTKPC']
    values = list(np.repeat(80*' ', len(keys)))
    comments = list(np.repeat(80*' ', len(keys)))
    #Now, they are populated and extended
    keys.extend(['FILT_' + str(i+1) for i in range(len(filters))])
    values[0] = 'The GRAMS O-rich grid for O-rich AGB and RSG stars'
    values[1] = 'Sargent, Srinivasan, & Meixner 2011 ApJ 728 93'
    values[2] = len(grid)
    values[3] = 'Kucinskas et al. (2005 A&A 442 281; 2006 A&A 452 1021), log(Z/Z_sun)=-0.5'
    values[4] = 'Oxygen-deficient silicates'
    values[5] = 'Ossenkopf et al. 1992 A&A 261 567'
    values[6] = 'KMH (Kim et al. 1994 ApJ 422 164) with (a_min, a_0) = (0.01, 0.1) \mu m'
    #If C-rich grid, change some of the above values
    if not(any('O-rich' in str(v) for v in values_orig)):
        values[0] = 'The GRAMS C-rich grid for C-rich AGB stars'
        values[1] = 'Srinivasan, Sargent, & Meixner 2011 A&A 532A 54'
        values[3] = 'Aringer et al. 2009 A&A 503 913'
        values[4] = 'Amorphous carbon and 10% by mass of SiC'
        values[5] = 'Zubko et al. 1996 MNRAS 282 1321, Pegourie 1988 A&A 194, 335'
        values[6] = 'KMH (Kim et al. 1994 ApJ 422 164) with (a_min, a_0) = (0.01, 1) \mu m'
    lmin = np.round(grid[0]['Lspec'].min(), decimals = 2); lmax = np.round(grid[0]['Lspec'].max(), decimals = 2)
    values[7] = '~' + str(lmin) + ' to ~' + str(lmax) + ' \mu m'
    values[8] = '50.12' #Hard-coded
    for f in filters:
        values.append('(' + f['filterName'] + ',' + str(np.round(f['lpivot'], decimals = 3)) + ')')
    comments[0] = ''
    comments[1] = 'Source for the grid'
    comments[2] = 'Number of models in the grid'
    comments[3] = 'Source for the photospheres'
    comments[4] = 'Type of dust'
    comments[5] = 'Source of dust opacities'
    comments[6] = 'Grain size distribution'
    comments[7] = 'Range of wavelengths in synthetic spectrum'
    comments[8] = 'Distance in kpc at which models are placed'
    for i in range(len(filters)):
        comments.append('Name/wavelength of filter #' + str(i+1))
    for i in range(len(keys)):
        h.append((keys[i], values[i], comments[i]), end = True)
    for i in range(loc[1]+1, len(header)):
        h.append((keys_orig[i], values_orig[i], comments_orig[i]), end = True)
    return h

def makegrid(infile = 'filters.csv', libraryFile = 'filters.hd5',
             outfile_suffix=''):
    """Compute GRAMS synthetic photometry in all the bands specified in infile, using the information
    from the filter library.
    INPUTS
       1) infile: a two-column CSV file of which the second column must contain
       must contain the names of the SVO/VOSA filter files to download. The first 
       column is not currently used, but can contain identifying information for 
       each filter that connects it back to the data.
       The filter names can be in the order of occurrence in the data, and may include
       repetitions (as it is quite possible that the data is compiled from a number of
       differing sets of observations).
       NOTE: this file must have a one-line header.

       2) libraryFile: the name of for the output hdf5 library.

       3) outfile_suffix: a string to append to the output grid file name.
    """
    filters_used = Table.read(infile, format = 'csv', names = ('column', 'filterName'))
    filterLibrary = pyp.get_library(fname = libraryFile)
    filterNames = [f['filterName'].replace('/','_') for f in filters_used]
    chemtype = ['o', 'c']
    #Links to the grid files on Google Drive. Is there a more elegant solution?
    file_link = {'o': 'https://ndownloader.figshare.com/files/9684331', \
                 'c': 'https://ndownloader.figshare.com/files/9684328'}
    for c in chemtype:
        gridfile = 'grams_' + c + outfile_suffix + '.fits'
        if os.path.isfile(gridfile):
            subprocess.call(['rm', gridfile])
        grid, header = fits.getdata(file_link[c], 1, header = True)
        #The original FITS_rec object is turned into an astropy Table for manipulation.
        #   It is then turned into a HDU object for output.
        grid = Table(grid) #conversion step 1
        print("Renaming 'MLR' column to 'DPR'")
        grid.rename_column('MLR', 'DPR') #Changing MLR column name to DPR
        inlam = grid[0]['Lspec']
        infnu = grid['Fspec']
        # infnu_star = grid['Fstar']

        _, seds = synthphot(inlam * units.um, infnu * units.Jy, filterLibrary, filterNames)

        # _, seds = pyp.extractSEDs(inlam, infnu, filters, Fnu=True, absFlux=False)
        # _, seds_star = pyp.extractSEDs(inlam, infnu_star, filters, Fnu=True, absFlux=False)
        # filters = filterLibrary.load_filters(filterNames, interp = True, lamb = inlam * pyp.unit['micron'])
        # The following ensures that the interpolated grid is high enough resolution
        #       because remember: the GRAMS grid has crappy resolution!
        lamb = np.geomspace(inlam.min(), inlam.max(), 1000) * pyp.unit['micron']
        filters = filterLibrary.load_filters(filterNames, interp = True, lamb = lamb)
        filters_used['lpivot'] = np.array([f.lpivot.magnitude for f in filters])
        del grid['Fphot']
        grid['Fphot'] = seds
        # grid['Fphot_star'] = seds_star
        #Update the magnitudes as well
        zp = np.array([f.Vega_zero_Jy.magnitude for f in filters])
        del grid['mphot']
        grid['mphot'] = -2.5 * np.log10(grid['Fphot'] / np.repeat(zp[np.newaxis, :], len(grid), axis = 0))
        g = fits.table_to_hdu(grid) #conversion step 2
        g.header = editgridheader(header, grid, filters_used)
        g.writeto(gridfile, overwrite = True)

def inspect_fits(data, fit, grid, prompt = '', outfile = 'out.csv', par_summary = True, **kwargs):
    """
    Interactively inspect a set of fits to a given SED.
    `prompt` and `outfile` set up the interactive inspection session.
    extra keyword arguments can contain multiple fit results, each of which will be compared to `fit` in the plot.
    Input: data - astropy table with data
        fit - astropy table with results from gramsfit
        grid - two-element dict such that grid['o'] is the astropy table with the O-rich GRAMS grid, and grid['c']
                contains the C-rich grid.
        prompt - not yet implemented
        outfile - not yet implemented
        kwargs - not yet implemented
    """
    ndata = len(data); n_models = len(fit[0]['modelindex_o'])
    distscale = (float(grid['o'].meta['DISTKPC']) / data['DKPC'])**2
    k = np.nonzero(['FILT_' in k for k in grid['o'].meta.keys()])[0]
    filternames = [f.split(',')[0].replace('(', '') for f in np.array(list(grid['o'].meta.values()))[k]]
    lpivot = np.array([float(f.split(',')[1].replace(')', '')) for f in np.array(list(grid['o'].meta.values()))[k]])

    plt = setPlotParams()
    plt.figure(figsize = (12, 12))
    color = {'o': 'blue', 'c': 'red'}
    xlim = [.1, 100]
    for i in range(ndata):
        ylim = np.nanmax(data[i]['FLUX'])
        chemtype = fit[i]['chemtype']
        modelindex = 'modelindex_' + chemtype
        scale = 'scale_' + chemtype
        # text = [r'$\chi^2 = {}$'.format(np.round(fit[i]['chisq_' + chemtype][0], decimals = 1)), \
        #         r'$\dot{M}_{\rm d}/{\rm M}_\odot~{\rm yr}^{-1} = {:0.1e}$'.format(fit[i]['DPR_' + chemtype]), \
        #         r'$L/{\rm L}_\odot = {:0.2e}$'.format(fit[i]['Lum_' + chemtype])]
        #Wrapper to ignore UserWarnings about converting Masked values to Nan.
        warnings.filterwarnings('ignore')
        title = 'ID = ' + str(fit[i]['ID']) + ', chemtype = ' + chemtype
        xscale = 'log'; yscale = 'log'
        xlabel = r'$\lambda (\mu$' + 'm)'; ylabel = r'$F_{\nu}$' + '(Jy)'
        if par_summary:
            fig, (a0, a1) = plt.subplots(2, 1, gridspec_kw = {'height_ratios': [3, 1]}, constrained_layout = True)
            a0.set_title(title)
            a0.set_xscale(xscale); a0.set_yscale(xscale)
            a0.set_xlabel(xlabel); a0.set_ylabel(ylabel)
            _ = a0.set_xlim(xlim)
            _ = a0.set_ylim(1e-5 * ylim, 1.2 * ylim)
        else:
            a0 = plt.copy()
            a0.title(title)
            a0.xscale(xscale); a0.yscale(xscale)
            a0.xlabel(xlabel); a0.ylabel(ylabel)
            _ = a0.xlim(xlim)
            _ = a0.ylim(1e-5 * ylim, 1.2 * ylim)
        for j in range(n_models):
            _ = a0.plot(grid[chemtype][fit[modelindex][i, 0]]['Lspec'], \
                        grid[chemtype][fit[modelindex][i, j]]['Fspec'] * fit[scale][i, j] * distscale[i], color = 'grey', alpha = 0.5)
        #Best fit model
        _ = a0.plot(grid[chemtype][fit[modelindex][i, 0]]['Lspec'], \
                    grid[chemtype][fit[modelindex][i, 0]]['Fspec'] * fit[scale][i, 0] * distscale[i], color = color[chemtype])
        #Alternate best fit models from kwargs
        for kw in kwargs:
            pass
        #Overlay data
        _ = a0.plot(lpivot[data[i]['BANDMAP']], data[i]['FLUX'], 'ko', linestyle = '')
        _ = a0.errorbar(lpivot[data[i]['BANDMAP']], data[i]['FLUX'], fmt = 'ko', yerr = data[i]['DFLUX'], linestyle = '')
        #Overlay text
        loc = [0.2, ylim * 1.1]
        # for i in range(len(text)):
        #     a0.text(loc[0], loc[1] / (i * 0.1 + 1), text[i])
        if par_summary:
            gramsfit.par_summary(a1, data[i], grid, fit[i], n_models = n_models)
            #fig.tight_layout()
            fig.show()
        else:
            plt.show()
    pass
