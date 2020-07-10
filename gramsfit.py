from astropy.table import Table, vstack
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
#import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import time

def chisq2d(scale, data, fmod):
    """Compute and return the chi-squared for an entire model grid,
    given the scale factors for each SED in the `data` table.
    """
    ndata, nmodels = scale.shape
    nbands = len(data[0]['FLUX'])
    chisqr = np.tile(np.nan, (ndata, nmodels))
    for i in range(ndata):
        detbands = np.where((data[i]['DETFLAG'])[data[i]['FITFLAG']])[0]
        ndetbands = np.where((~data[i]['DETFLAG'])[data[i]['FITFLAG']])[0]
        f = np.tile((data[i]['FLUX'])[:, np.newaxis], nmodels)
        df = np.tile((data[i]['DFLUX'])[:, np.newaxis], nmodels)
        sc = np.tile(scale[i, :][:, np.newaxis], nbands).transpose()
        residue = (f - sc * fmod[:, data[i]['BANDMAP']].transpose()) / df
        #The following chisq has two terms -- the first one for detections,
        #   the second for non-detections.
        #Two differences between the following and Sawicki et al. 2012, especially Eq. A10.
        #   1) He omitted one sigma_j in the proportionality in Eq. A6, which cancels
        #       when the substitution is made for the error function.
        #   2) I've used norm.cdf instead of the error function.
        chisqr[i, :] = np.nansum(residue[detbands]**2, axis = 0)
        if len(ndetbands) > 0:
            chisqr[i, :] = chisqr[i, :] - 2 * np.nansum(np.log(norm.cdf(residue[ndetbands])), axis = 0)
    return chisqr

def chisq1d(s, data, fmod):
    """return chisq for one source and one model and one scale factor"""
    fm = fmod[data['BANDMAP'][data['FITFLAG']]]
    f = data['FLUX'][data['FITFLAG']]
    df = data['DFLUX'][data['FITFLAG']]
    residue = (f - s * fm)/df
    #
    #The following chisq has two terms -- the first one for detections,
    #   the second for non-detections.
    #Two differences between the following and Sawicki et al. 2012 PASP 124, 1208, especially Eq. A10.
    #   1) He omitted one sigma_j in the proportionality in Eq. A6, which cancels
    #       when the substitution is made for the error function.
    #   2) I've used norm.cdf instead of the error function.
    chisq = np.nansum(residue[data['DETFLAG'][data['FITFLAG']]]**2) - 2 * \
        np.nansum(np.log(norm.cdf(residue[~data['DETFLAG'][data['FITFLAG']]])))

    return chisq

def get_scale(data, fmod):
    """Compute the scale factor such that the sum of (fdata - scale * fmod)**2 / (dfdata)**2 
    over all valid bands is minimised. Can also be computed with non-detections in the data 
    (Method from Sawicki et al. 2012 PASP 124, 1208).
    Input: data - ndata-element astropy table with nbands bands of data per source. Must include
                    a column FITFLAG that is True for each band to be included in the fit.
           fmod - n_accept-by-nbands element array of fluxes for each model in the O- or C-rich GRAMS grid.
                    n_accept is the number of models with the lowest chi-squares whose fits are deemed acceptable.
                    This trimming is applied when computing the chi-squares in get_chisq.
    Output: scale - ndata-by-n_accept array with computed scale values.
    """
    n_models = fmod.shape[0]
    ndata, nbands = data['FLUX'].shape
    #First, obtain the scale factor ignoring any non-detection
    scale_det = np.tile(np.nan, (ndata, n_models))
    for i in range(ndata):
        bands = np.nonzero((data[i]['DETFLAG']) & (data[i]['FITFLAG']))[0]
        f = (np.tile(data[i]['FLUX'][bands][:, np.newaxis], n_models)).T
        df = (np.tile(data[i]['DFLUX'][bands][:, np.newaxis], n_models)).T
        with np.errstate(divide='ignore',invalid='ignore'):
            scale_det[i, :] = np.nansum(f * fmod[:, bands] / df**2, axis = 1) / np.nansum(fmod[:, bands]**2 / df**2, axis = 1)
    scale_nondet = scale_det.copy()
    #Only required if there are non-detections in the data.
    k = np.nonzero([not(data[j]['DETFLAG'][data[j]['FITFLAG']].any()) for j in range(ndata)])[0]
    if len(k) > 0:
        for i in range(len(k)):
            for j in range(n_models):
                p = minimize(chisq1d, np.array([scale_det[i, j]]), args = (data[i, :], fmod[j, :]), \
                             method = 'Nelder-Mead', options = {'maxiter': 10000})
                if p['status'] == 0:
                    scale_nondet[i, j] = p['x'][0]
                else:
                    print("Search for scale_nondet did not converge for object " + \
                          "{}.".format(data['ID']))
    return scale_nondet

def get_chisq(data, ofmod, cfmod, n_accept = 100, scale = False):
    """Given the observed photometry and the model grid synthetic photometry,
    compute and output the chisq with and without the scale as a free parameter.
    Input: data - ndata-element astropy table with nbands bands of data per source. Must include
                    a column FITFLAG that is True for each band to be included in the fit.
           [o/c]fmod - nmodels-by-nbands array of fluxes for each model in the O- or C-rich GRAMS grid.
           n_accept - the number of models with the lowest chi-squares whose fits are deemed acceptable.
           scale - True or False depending on whether the scale is treated as a free parameter.
    Output: chisq_o, chisq_c - ndata-by-n_accept array of chi-squares for the n_accept models of
                    each chemical type with the lowest chi-squares.
            modelindex_o, modelindex_c - ndata-by-n_accept arrays of indices into the GRAMS grid of the 
                    n_accept models with the lowest chi-squares for each source.
            scale_o, scale_c - ndata-by-n_accept arrays of scale factors corresponding to the models with
                    the lowest chi-squares.
    """
    ndata = len(data)
    #Number of free parameters increased by one if scale is provided.
    pp = 0
    #number of bands with FITFLAG == True that have finite fluxes
    nfinite = np.array([len(np.nonzero(~np.isnan(x['FLUX'][x['FITFLAG']]))[0]) for x in data])
    with np.errstate(divide='ignore',invalid='ignore'):
        if scale:
            scale_o = get_scale(data, ofmod)
            scale_c = get_scale(data, cfmod)
            chisq_o = chisq2d(scale_o, data, ofmod) / (np.tile(nfinite[:, np.newaxis], len(ofmod)) - pp)
            chisq_c = chisq2d(scale_c, data, cfmod) / (np.tile(nfinite[:, np.newaxis], len(cfmod)) - pp)
        else:
            scale_o = np.tile(1.0, (ndata, len(ofmod)))
            scale_c = np.tile(1.0, (ndata, len(cfmod)))
            chisq_o = chisq2d(scale_o, data, ofmod) / (np.tile(nfinite[:, np.newaxis], len(ofmod)) - pp)
            chisq_c = chisq2d(scale_c, data, cfmod) / (np.tile(nfinite[:, np.newaxis], len(cfmod)) - pp)
    #for each source, sort according to increase chisq. This sorted index into the model grid
    #   is stored in the modelindex arrays for each grid, and is output along with the sorted
    #   chisq values.
    #Only the first n_accept models are retained.
    modelindex_o = np.argsort(chisq_o, axis = 1)[:, :n_accept]
    modelindex_c = np.argsort(chisq_c, axis = 1)[:, :n_accept]
    chisq_o = np.array([chisq_o[i, modelindex_o[i, :]] for i in range(ndata)]).copy()
    chisq_c = np.array([chisq_c[i, modelindex_c[i, :]] for i in range(ndata)]).copy()
    scale_o = np.array([scale_o[i, modelindex_o[i, :]] for i in range(ndata)]).copy()
    scale_c = np.array([scale_c[i, modelindex_c[i, :]] for i in range(ndata)]).copy()
    return chisq_o, chisq_c, modelindex_o, modelindex_c, scale_o, scale_c

def get_chemtype(chisq_o_min, chisq_c_min, CFIT_TOL = 1.0):
    """Given the minimum chisq values for each chemical type for each source, return
    the chemical type.
    CFIT_TOL is the tolerance defined such that the C-rich chisq has to be lower than
    the O-rich chisq divided by CFIT_TOL.
    """
    chem = np.array(['o', 'c'])
    return chem[(chisq_c_min <= chisq_o_min / CFIT_TOL) * 1]

def par_summary(plt, data, grid, fit, n_models = 100):
    """Boxplot summaries for the best-fit models of both chemistries to a single source.
    For each chemistry, the n_models models with the lowest chisq values are selected.
    For this to be true, the modelindex and scale arrays must already be sorted 
    in ascending order of chisq.
    plt: plot object instantiated before being passed into this module.
    data: 1-row astropy table. 
    modelindex_, scale_: 1 x ngrid numpy arrays where ngrid is 
        the number of models in that grid.
    chemtype: 1-element array ('o' or 'c') corresponding to given source.
    ogrid, cgrid: the full grid of models for both chemical types.
    """
    def draw_plot(plt, data, edge_color, fill_color, data_labels = None):
        bp = plt.boxplot(data, labels = data_labels, patch_artist=True, \
                         meanline = True, showmeans = True)
        #for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        #    plt.setp(bp[element], color=edge_color)
        for patch in bp['boxes']:
            patch.set(facecolor=fill_color)
        yt = plt.get_yticks(); plt.set_yticks(np.linspace(yt[0], yt[-1], 8))
        return bp
    lamref = {'o': '10', 'c': '11.3'}
    #One boxplot for each chemical type
    color = {'o': 'blue', 'c': 'red'}
    for t in ['o', 'c']:
        g = grid[t][fit['modelindex_' + t]][0:n_models]
        d = [fit['scale_' + t][0:n_models] * g['Lum'] / 1e3, g['Teff'] / 1e3, np.log10(g['Rin']), \
             -np.log10(g['tau1']), -np.log10(g['tau' + lamref[t].replace('.', '_')]), \
             np.log10(np.sqrt(fit['scale_' + t][0:n_models]) * g['DPR'] / 1e-13), \
             g['Tin'] / 1e3, np.log10(fit['scale_' + t][0:n_models])]
        d_labels = [r"$\bm{L/10^3 \textbf{\textrm{L}}}_\odot$", \
                    r"$\bm{T_\textbf{\textrm{eff}}/10^3 \textbf{\textrm{K}}}$", \
                    #r"$\bm{\log{(T_\textbf{\textrm{eff}}/10^3 \textbf{\textrm{K}})}}$", \
                    r"$\bm{\log{(R_\textbf{\textrm{in}}/R_\textbf{\textrm{star}})}}$", \
                    r"$\bm{-\log{\tau_1}}$", r"$\bm{-\log{\tau_{" + lamref[t] + "}}}$", \
                    r"$\bm{\log{(\textbf{\textrm{DPR}}/10^{-13} \textbf{\textrm{M}}_\odot \textbf{\textrm{yr}}^{-1})}}$", \
                    r"$\bm{T_\textbf{\textrm{eff}}/10^3 \textbf{\textrm{K}}}$", r"$\bm{\log{s}}$"]
        bp = draw_plot(plt, d, color[t], 'None', data_labels = d_labels)
    #plt.xticks(fontsize = 8)
    for tick in plt.xaxis.get_major_ticks():
        tick.label.set_fontsize(8)
    #plt.set_title("gramsfit parameter summary ({} best-fit models) for source {} (chemtype = {})".\
    #          format(n_models, data['ID'], fit['chemtype']), fontsize = 12)
    plt.set_title("Parameter summary ({} best-fit models)".format(n_models))

def get_pars(fit, ogrid, cgrid):
    """Given the fit table and the model grids, return the best-fit parameter values and estimates of their
    uncertainties.
    The fit table has one row per source, and n_accept models per chemical type per source. This filtering
    is already done in get_chisq.
    All n_accept models are used to compute the MADM.
    """
    def madm(par):
        result = np.nanmedian(np.abs(par - np.nanmedian(par)))
        if result == 0:
            return np.nan
        else:
            return result
    #
    ndata, nmodels = fit['chisq_o'].shape

    parnames = np.array(['Lum', 'Teff', 'logg', 'Rin', 'tau1', 'DPR', 'Tin']) #common columns between chemical types
    npars = len(parnames)
    nanarray = np.tile(np.nan, ndata)
    l = []
    for name in parnames:
        l.append(nanarray)
    po = Table(l, names = tuple(parnames), dtype=tuple(np.repeat('f8', len(parnames)))); po_err = po.copy()
    #po = Table(l, names = tuple(parnames), dtype=('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8')); po_err = po.copy()
    l = 0
    #add a scale column
    po['scale'] = np.repeat(nanarray[:, np.newaxis], nmodels, axis = 1); po_err['scale'] = nanarray
    #similar tables for C-rich chemistry
    pc = po.copy(); pc_err = po_err.copy()
    #add columns for optical depth at feature centre
    po['tau10'] = nanarray; po_err['tau10'] = nanarray
    pc['tau11_3'] = nanarray; pc_err['tau11_3'] = nanarray
    for i in range(ndata):
        for name in parnames:
            po[i][name] = ogrid[fit[i]['modelindex_o']][name][0]
            po_err[i][name] = madm(ogrid[fit[i]['modelindex_o']][name])
        po[i]['tau10'] = ogrid[fit[i]['modelindex_o']]['tau10'][0]
        po_err[i]['tau10'] = madm(ogrid[fit[i]['modelindex_o']]['tau10'])
        po[i]['scale'] = fit[i]['scale_o']
        po_err[i]['scale'] = madm(po[i]['scale'])
        po[i]['Lum'] *= po[i]['scale'][0]
        po_err[i]['Lum'] *= po[i]['scale'][0]
        po[i]['DPR'] *= np.sqrt(po[i]['scale'][0])
        po_err[i]['DPR'] *= np.sqrt(po[i]['scale'][0])
        for name in parnames:
            pc[i][name] = cgrid[fit[i]['modelindex_c']][name][0]
            pc_err[i][name] = madm(cgrid[fit[i]['modelindex_c']][name])
        pc[i]['tau11_3'] = cgrid[fit[i]['modelindex_c']]['tau11_3'][0]
        pc_err[i]['tau11_3'] = madm(cgrid[fit[i]['modelindex_c']]['tau11_3'])
        pc[i]['scale'] = fit[i]['scale_c']
        pc_err[i]['scale'] = madm(pc[i]['scale'])
        pc[i]['Lum'] *= pc[i]['scale'][0]
        pc_err[i]['Lum'] *= pc[i]['scale'][0]
        pc[i]['DPR'] *= np.sqrt(pc[i]['scale'][0])
        pc_err[i]['DPR'] *= np.sqrt(pc[i]['scale'][0])
    return po, po_err, pc, pc_err

def prep_input(data, ogrid, cgrid, ID = None, FITFLAG = None, DKPC = None):
    #data, ogrid, and cgrid can either be a string pointing to the full path of the file,
    #   or an astropy table
    for x, name in zip([data, ogrid, cgrid], ['data', 'ogrid', 'cgrid']):
        if isinstance(x, str):
            if 'vot' in x:
                form = 'votable'
            elif 'fits' in x:
                form = 'fits'
            else:
                raise ValueError("Input file {} does not have .vot or .fits extension!".format(x))
            exec(name + " = Table.read('" + x + "', format = '" + form + "')")
    ndata = len(data)
    #If the ID column is absent from the data, generate unique IDs from indices.
    if 'ID' not in data.columns:
        data['ID'] = [str(i+1) for i in np.arange(len(data))]
    #If FITFLAG and/or DKPC keywords are provided, override values in the data table.
    if FITFLAG is not None:
        if FITFLAG.ndim == 1:
            data['FITFLAG'] = np.repeat(FITFLAG[:, np.newaxis], ndata, axis = 1)
        else:
            data['FITFLAG'] = FITFLAG
    if DKPC is not None:
        if hasattr(DKPC, '__len__'):
            data['DKPC'] = DKPC
        else:
            data['DKPC'] = np.repeat(DKPC, ndata)
    #Scale data fluxes to LMC distance (distance at which models are computed)
    try:
        modeldkpc = float(ogrid.meta['DISTKPC'])
    except:
        modeldkpc = 50.12
    distscale = (np.repeat(data['DKPC'][:, np.newaxis], data['FLUX'].shape[1], axis = 1)/modeldkpc)**2
    data['FLUX'] *= distscale
    data['DFLUX'] *= distscale

    return data, ogrid, cgrid

def gramsfit(data, ogrid, cgrid, ID = None, FITFLAG = None, DKPC = None, scale = False, \
             force_chemtype = None, n_accept = 100, compute_pars = True):
    """
    Compute chi-squared fits to observed SEDs of AGB/RSG candidates using the GRAMS O-rich and C-rich
    model grids.
    INPUT:
    data - Astropy table containing the observed fluxes and uncertainties over a number of broadband filters.
        Each row of the data table must contain the following columns:
        ID - unique identifier for each SED.
        FLUX/DFLUX - NBANDS-element arrays containing the observed fluxes and uncertainties in each of
                    NBANDS broadband filters.
        BANDMAP - NBANDS-element arrays of indices into the array of broadband filters
        DETFLAG - NBANDS-element arrays of Booleans, TRUE if the flux in that band is a detection limit.
                    In such a case, the value in DFLUX is taken to be the number of standard deviations
                    above the noise level of this detection limit.
        OPTIONAL COLUMNS:
        FITFLAG - NBANDS-element arrays of Booleans, TRUE if band is to be included in the fit.
                    Can also be fed as an input keyword to the function call.
        DKPC - distance in kpc to source. Can also be fed as input keyword to the function call.
    ogrid, cgrid - Astropy tables containing synthetic photometry for the GRAMS grid over a number of 
        broadband filters. Photometry can be computed from the synthetic spectra for any broadband filters 
        present in the SVO database: http://svo2.cab.inta-csic.es/theory/fps/
    OPTIONAL INPUT:
    ID - array of unique identifiers passed to the function. Overrides column present in the data table.
    FITFLAG - NBANDS-element arrays of Booleans, TRUE if the flux in that band is a detection limit. Overrides column
            present in the data table.
    DKPC - distance in kpc (scalar or array) passed to the function. Overrides column present in the data table.
    scale - Boolean. If TRUE, a best-fit luminosity scale factor is also computed as part of the fit. The
            effective distance to the source is then DKPC_eff = data['DKPC'] / np.sqrt(scale).
            See the get_scale method for details.
    force_chemtype - scalar or array containing either 'o' or 'c', overrides the best-fit chemical type computed
            from the chi-squared fit.
    n_accept - scalar, number of models with lowest chi-squares used to compute the parameter uncertainties.
    compute_pars - Boolean. If TRUE, best-fit parameter values and related uncertainties are computed using
            the specified n_accept value.
    OUTPUT:
    fit - Astropy table containing the following columns for each input SED:
        chisq_o, chisq_c - n_accept-element arrays with the lowest n_accept chi-squared values computed for each 
        chemical type.
        chemtype - chemical type assigned based on comparing the lowest chi-squared values of each chemical type.
        modelindex_o, modelindex_c - n_accept-element arrays with indices into the model grids for the models with 
                the lowest chi-squared values for each chemical type.
        scale_o, scale_c - n_accept-element arrays containing best-fit luminosity scale factors for each of the 
                n_accept models with the lowest chi-squared values for each chemical type.
        If compute_pars is True, best-fit values and uncertainties (from the n_accept models with lowest 
                chi-squared values) are computed for the following parameters for each chemical type:
                Lum, Teff, logg, Rin, tau1, DPR, Tin, scale, and tau10 (O-rich) or tau11_3 (C-rich).
    """
    #In case the user has input the table names instead of the tables, read in the tables.
    #Also, scale the data fluxes to the distace at which the models are computed.
    d = data.copy(); og = ogrid.copy(); cg = cgrid.copy()
    data, ogrid, cgrid = prep_input(d, og, cg, ID = ID, FITFLAG = FITFLAG, DKPC = DKPC)
    d = 0; og = 0; cg = 0
    #computing chi-squares
    if scale:
        chisq_o, chisq_c, modelindex_o, modelindex_c, scale_o, scale_c = \
            get_chisq(data, ogrid['Fphot'], cgrid['Fphot'], n_accept = n_accept, scale = True)
    else:
        chisq_o, chisq_c, modelindex_o, modelindex_c, scale_o, scale_c = \
            get_chisq(data, ogrid['Fphot'], cgrid['Fphot'], n_accept = n_accept, scale = scale)
    #set chemical types
    chemtype = get_chemtype(chisq_o[:, 0], chisq_c[:, 0])
    #scale data fluxes back to data['DKPC'] values, create a table to store output
    data['FLUX'] /= distscale
    data['DFLUX'] /= distscale
    fit = Table([data['ID'], chemtype, chisq_o, chisq_c, modelindex_o, modelindex_c, scale_o, scale_c], \
                 names = ('ID', 'chemtype', 'chisq_o', 'chisq_c', 'modelindex_o', 'modelindex_c', 'scale_o', 'scale_c'))
    #Compute best-fit parameter values
    if compute_pars:
        po, po_err, pc, pc_err = get_pars(fit, ogrid, cgrid)
        for c in po.columns:
            fit[c + '_o'] = po[c]
            fit[c + '_o_err'] = po_err[c]
        for c in pc.columns:
            fit[c + '_c'] = pc[c]
            fit[c + '_c_err'] = pc_err[c]
        print("...done.")
    #Force chemical types
    if force_chemtype is not None:
        nforce = len(force_chemtype)
        if nforce == 1:
            force_chemtype = np.repeat(force_chemtype, ndata).copy()
        ct = np.array([f.strip().lower() for f in list(force_chemtype)])
        if ((ct == 'o') & (ct == 'c')).sum() < len(ct):
            raise ValueError("ERROR! Chemical type can only be either 'O' or 'C'!")
        else:
            print("Forcing chemical types as specified by force_chemtype...")
            fit.chemtype = ct

    return fit

def deploy_gramsfit(data, ogrid, cgrid, scale):
    """Internal function only to be accessed via gramsfit_driver.
    """
    print("depoly_gramsfit: executing gramsfit for {} sources.".format(len(data)))
    fit = gramsfit(data, ogrid, cgrid, scale = scale)
    #Before saving to file, remove all columns related to the parameter values.
    #TBD: save the parameters as part of the table but in a different format (inside of a table inside a table)
    #       This has to be done within gramsfit.get_pars.
    cols = [col for col in fit.columns if 'pars' in col]
    if len(cols) != 0:
        for col in cols:
            del fit[col]
    return fit

def gramsfit_wrapper(data, ogrid, cgrid, scale = False, outfile = 'gramsfit.vot', parallel = False, n_cores = 3):
    """Use for computationally intensive/large datasets. This method splits the data into chunks and also allows
    parallel calls to the gramsfit method for these chunks.
    The data is first split into chunks with <50,000 rows each, then each chunk is fed to gramsfit.
    If parallel is set to True, each chunk is further split into n_cores pieces, followed by parallel calls to gramsfit for each piece.
    The results are automatically combined and written into the output file.
    Input: data, ogrid, cgrid - MUST be astropy tables.
        scale - Boolean. If True, luminosity scaling is incorporated into the fits. See the get_scale method for details.
        outfile - results are dumped into a VOTable with this name. Defaults to 'gramsfit.vot'
        parallel - Boolean. If True, the data is further split across n_cores simultaneous calls to gramsfit.
        n_cores - number of CPU cores over which parallel calls are to be executed.
    """
    ndata = len(data)
    #Depending on the data size, decide whether the payload must be (1) split and (2) executed using multiprocess.
    n_max_per_piece = 50000
    if ndata <= n_max_per_piece:
        datalist = [data]
    else:
        n_pieces = int(np.ceil(ndata / n_max_per_piece))
        datalist = [data[i * n_max_per_piece: min([n_max_per_piece * (i + 1), ndata])] for i in range(n_pieces)]
    fits = []
    for i in range(len(datalist)):
        if parallel & (cpu_count() >= n_cores):
            n_max_per_core = int(np.ceil(len(datalist[i]) / n_cores))
            values = [(datalist[i][j * n_max_per_core: min([n_max_per_core * (j + 1), len(datalist[i])])], \
                       ogrid, cgrid, scale) for j in range(n_cores)]
            start = time.perf_counter()
            with Pool(processes = n_cores) as pool:
                result = pool.starmap(deploy_gramsfit, tuple(values))
            finish = time.perf_counter()
            print("Parallel execution finished in {} second(s).".format(round(finish-start, 2)))
        else:
            print("Either parallelisation was not chosen, or the number of CPU cores < n_cores. Running in serial mode.")
            result = deploy_gramsfit(datalist[i], ogrid, cgrid, scale = scale)
        if len(result) > 1:
            fit = vstack(result)
        else:
            fit = result[0]
        fits.append(fit)
    if len(datalist) == 1:
        f = fits[0].copy()
    else:
        f = vstack(fits)
    #Output
    print("Output written to {}.".format(outfile))
    f.write(outfile, format = 'votable', overwrite = True)
