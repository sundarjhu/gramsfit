from astropy.table import Table
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from gramsfit_utils import *

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
    #Two differences between the following and Sawicki et al. 2012, especially Eq. A10.
    #   1) He omitted one sigma_j in the proportionality in Eq. A6, which cancels
    #       when the substitution is made for the error function.
    #   2) I've used norm.cdf instead of the error function.
    chisq = np.nansum(residue[data['DETFLAG'][data['FITFLAG']]]**2) - 2 * \
        np.nansum(np.log(norm.cdf(residue[~data['DETFLAG'][data['FITFLAG']]])))

    return chisq

def get_scale(data, fmod):
    n_models = fmod.shape[0]
    ndata, nbands = data['FLUX'].shape
    #First, obtain the scale factor ignoring any non-detection
    scale_det = np.tile(np.nan, (ndata, n_models))
    for i in range(n_models):
        bands = (data[i]['DETFLAG'])[data[i]['FITFLAG']]
        f = np.tile(data[i]['FLUX'][bands][:, np.newaxis], n_models)
        df = np.tile(data[i]['DFLUX'][bands][:, np.newaxis], n_models)
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

def get_chisq(data, ofmod, cfmod, scale = False):
    """Given the observed photometry and the model grid synthetic photometry,
    compute and output the chisq with and without the scale as a free parameter."""
    ndata = len(data)
    #Number of free parameters increased by one if scale is provided.
    pp = 0
    #number of bands with FITFLAG == True that have finite fluxes
    nfinite = np.array([len(np.nonzero(~np.isnan(x['FLUX'][x['FITFLAG']]))[0]) for x in data])
    if scale:
        s_o = get_scale(data, ofmod)
        s_c = get_scale(data, cfmod)
        c_o = chisq2d(s_o, data, ofmod) / (np.tile(nfinite[:, np.newaxis], len(ofmod)) - pp)
        c_c = chisq2d(s_c, data, cfmod) / (np.tile(nfinite[:, np.newaxis], len(cfmod)) - pp)
    else:
        s_o = np.tile(1.0, (ndata, len(ofmod)))
        s_c = np.tile(1.0, (ndata, len(cfmod)))
        c_o = chisq2d(s_o, data, ofmod) / (np.tile(nfinite[:, np.newaxis], len(ofmod)) - pp)
        c_c = chisq2d(s_c, data, cfmod) / (np.tile(nfinite[:, np.newaxis], len(cfmod)) - pp)
    #for each source, sort according to increase chisq. This sorted index into the model grid
    #   is stored in the modelindex arrays for each grid, and is output along with the sorted
    #   chisq values.
    modelindex_o = np.argsort(c_o, axis = 1)
    modelindex_c = np.argsort(c_c, axis = 1)
    chisq_o = np.array([c_o[i, modelindex_o[i, :]] for i in range(ndata)])
    chisq_c = np.array([c_c[i, modelindex_c[i, :]] for i in range(ndata)])
    scale_o = np.array([s_o[i, modelindex_o[i, :]] for i in range(ndata)])
    scale_c = np.array([s_c[i, modelindex_c[i, :]] for i in range(ndata)])
    return chisq_o, chisq_c, modelindex_o, modelindex_c, scale_o, scale_c

def get_chemtype(chisq_o, chisq_c, CFIT_TOL = 1.0):
    """Given arrays chisq_o and chisq_c of shape (Ndata x Nmodels), return the chemical type."""
    """CFIT_TOL is the tolerance defined such that the C-rich chisq has to be lower than
    the O-rich chisq divided by CFIT_TOL."""
    chems = np.array(['o', 'c'])
    ndata = chisq_o.shape[0]
    chemtype = np.array([chems[(chisq_c[i, 0] <= \
                                chisq_o[i, 0]/CFIT_TOL)*1] for i in range(ndata)])
    return chemtype

def par_summary(plt, data, grid, fit, n_models = 100):
    """Boxplot summaries for the best-fit models of both chemistries to a single source.
    For each chemistry, the n_models models with the lowest chisq values are selected.
    For this to be true, the modelindex, scale, and flag arrays must already be sorted 
    in ascending order of chisq.
    plt: plot object instantiated before being passed into this module.
    data: 1-row astropy table. 
    modelindex_, scale_, flag_: 1 x ngrid numpy arrays where ngrid is 
        the number of models in that grid.
    chemtype: 1-element array ('o' or 'c') corresponding to given source.
    ogrid, cgrid: the full grid of models for both chemical types."""
    #plt.figure(figsize = (12, 3))
    def draw_plot(plt, data, edge_color, fill_color, data_labels = None):
        bp = plt.boxplot(data, labels = data_labels, patch_artist=True, \
                         meanline = True, showmeans = True)
        #for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        #    plt.setp(bp[element], color=edge_color)
        for patch in bp['boxes']:
            patch.set(facecolor=fill_color)
        return bp
    lamref = {'o': '10', 'c': '11.3'}
    #One boxplot for each chemical type
    for t in ['o', 'c']:
        g = grid[t][fit['modelindex_' + t]][0:n_models]
        d = [np.log10(fit['scale_' + t][0:n_models] * g['Lum'] / 1e3), np.log10(g['Teff'] / 1e3), np.log10(g['Rin']), \
             -np.log10(g['tau1']), -np.log10(g['tau' + lamref[t].replace('.', '_')]), \
             np.log10(np.sqrt(fit['scale_' + t][fit['flag_' + t]][0:n_models]) * g['MLR'] / 1e-13), \
             np.log10(g['Tin'] / 1e3), np.log10(fit['scale_' + t][fit['flag_' + t]][0:n_models])]
        d_labels = [r"$\bm{\log{(L/10^3} \textbf{\textrm{L}}_\bm{\odot})}$", \
                    r"$\bm{\log{(T_\textbf{\textrm{eff}}/10^3 \textbf{\textrm{K}})}}$", \
                    r"$\bm{\log{(R_\textbf{\textrm{in}}/R_\textbf{\textrm{star}})}}$", \
                    r"$\bm{-\log{\tau_1}}$", r"$\bm{-\log{\tau_{" + lamref[t] + "}}}$", \
                    r"$\bm{\log{(\textbf{\textrm{DPR}}/10^{-13} \textbf{\textrm{M}}_\odot \textbf{\textrm{yr}}^{-1})}}$", \
                    r"$\bm{\log{(T_\textbf{\textrm{in}}/10^3 \textbf{\textrm{K}})}}$", r"$\bm{\log{s}}$"]
        bp = draw_plot(plt, d, 'blue', 'None', data_labels = d_labels)
    #plt.xticks(fontsize = 8)
    for tick in plt.xaxis.get_major_ticks():
        tick.label.set_fontsize(8)
    #plt.set_title("gramsfit parameter summary ({} best-fit models) for source {} (chemtype = {})".\
    #          format(n_models, data['ID'], fit['chemtype']), fontsize = 12)
    plt.set_title("Parameter summary ({} best-fit models)".format(n_models))

def get_pars(grid, chisq, modelindex, scale, flag, n_accept = 100):
    """Given the chi-square and indices into the model grid, return the 
        best-fit values and uncertainty estimates for the parameters.
        chisq and modelindex must have shapes (Ndata x Nmodels), where
        Nmodels is the number of models in grid.
        n_accept is the number of models to consider when
        computing parameter uncertainties (by default, the 100 models
        with the lowest chisq for each source are considered).
    """
    def madm(par):
        result = np.median(np.abs(par - np.median(par)))
        if result == 0:
            return np.nan
        else:
            return result
    #
    ndata, nmodels = chisq.shape
    if nmodels != len(grid):
        raise ValueError("get_pars: shape of chisq does not match number of models in grid!")
    if 'tau10' in grid.columns:
        taulabel = 'tau10'
    else:
        taulabel = 'tau11_3'
    parnames = np.array(['Lum', 'Teff', 'logg', 'Rin', 'tau1', taulabel, 'MLR', 'Tin'])
    npars = len(parnames)
    nanarray = np.tile(np.nan, ndata)
    l = []
    for name in parnames:
        l.append(nanarray)
    p = Table(l, names = tuple(parnames), dtype=('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8'))
    p_err = p.copy()
    #add a scale column
    p['scale'] = nanarray
    p_err['scale'] = nanarray
    for i in range(ndata):
        try:
            for name in parnames:
                p[i][name] = grid[modelindex[i, flag[i, :]]][name][0]
                p_err[i][name] = madm(grid[modelindex[i, flag[i, :]]][name][0:n_accept])
            p[i]['scale'] = scale[i, modelindex[i, flag[i, :]]][0]
            p[i]['Lum'] *= p[i]['scale']
            p_err[i]['Lum'] *= p[i]['scale']
            p[i]['MLR'] *= np.sqrt(p[i]['scale'])
            p_err[i]['MLR'] *= np.sqrt(p[i]['scale'])
        except:
            pass
    return p, p_err

def gramsfit(data, ogrid, cgrid, ID = None, FITFLAG = None, DKPC = None, scale = False, \
             force_chemtype = None, n_accept = 100):
    #data, ogrid, and cgrid can either be a string pointing to the full path of the file,
    #   or an astropy table
    if isinstance(data, str):
        if 'vot' in data:
            form = 'vot'
        else:
            form = 'fits'
        d = Table.read(data, format = form)
        data = d.copy()
        d = 0.
    if isinstance(ogrid, str):
        if 'vot' in data:
            form = 'vot'
        else:
            form = 'fits'
        d = Table.read(ogrid, format = form)
        ogrid = d.copy()
        d = 0.
    if isinstance(cgrid, str):
        if 'vot' in data:
            form = 'vot'
        else:
            form = 'fits'
        d = Table.read(cgrid, format = form)
        cgrid = d.copy()
        d = 0.
    #
    ndata = len(data)
    if 'ID' not in data.columns:
        data['ID'] = [str(i+1) for i in np.arange(len(data))]
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

    print("gramsfit: computing chi-squares...")
    if scale:
        chisq_o, chisq_c, modelindex_o, modelindex_c, scale_o, scale_c = \
            get_chisq(data, ogrid['Fphot'], cgrid['Fphot'], scale = True)
    else:
        chisq_o, chisq_c, modelindex_o, modelindex_c, scale_o, scale_c = \
            get_chisq(data, ogrid['Fphot'], cgrid['Fphot'], scale = scale)
    print("gramsfit: done computing chi-squares.")
    print("gramsfit: determining chemical types...")
    chemtype = get_chemtype(chisq_o, chisq_c)
    print("gramsfit: done computing chemical types.")
    #recover original data
    data['FLUX'] /= distscale
    data['DFLUX'] /= distscale
    #
    #restrict the values of scale such that the scaled luminosities lie within the luminosity range
    #   of each grid
    #This could be changed for the high-luminosity range to treat high-L sources.
    #This should really be done at the optimisation stage by placing constraints on the scale,
    #   but for now...
    flag_o = np.array([(scale_o[i, :] * ogrid[modelindex_o[i, :]]['Lum'] >= ogrid['Lum'].min()) & \
                   (scale_o[i, :] * ogrid[modelindex_o[i, :]]['Lum'] <= ogrid['Lum'].max()) \
                   for i in range(len(data))])
    flag_c = np.array([(scale_c[i, :] * cgrid[modelindex_c[i, :]]['Lum'] >= cgrid['Lum'].min()) & \
                   (scale_c[i, :] * cgrid[modelindex_c[i, :]]['Lum'] <= cgrid['Lum'].max()) \
                   for i in range(len(data))])
    #Best-fit parameter values
    print("gramsfit: computing best-fit parameter values...")
    po, po_err = get_pars(ogrid, chisq_o, modelindex_o, scale_o, flag_o, n_accept = n_accept)
    pc, pc_err = get_pars(cgrid, chisq_c, modelindex_c, scale_c, flag_c, n_accept = n_accept)
    print("gramsfit: done computing best-fit parameter values.")

    fit = Table([data['ID'], chemtype, chisq_o[:, :n_accept], chisq_c[:, :n_accept], \
                 modelindex_o[:, :n_accept], modelindex_c[:, :n_accept], \
                 scale_o[:, :n_accept], scale_c[:, :n_accept], flag_o[:, :n_accept], flag_c[:, :n_accept], \
                 po, po_err, pc, pc_err], \
                names = ('ID', 'chemtype', 'chisq_o', 'chisq_c', 'modelindex_o', \
                         'modelindex_c', 'scale_o', 'scale_c', 'flag_o', 'flag_c', \
                         'pars_o', 'pars_o_err', 'pars_c', 'pars_c_err'))

    if force_chemtype is not None:
        nforce = len(force_chemtype)
        if nforce == 1:
            force_chemtype = np.repeat(force_chemtype, ndata)
        ct = np.array([f.strip().lower() for f in list(force_chemtype)])
        if ((ct == 'o') & (ct == 'c')).sum() < len(ct):
            raise ValueError("ERROR! Chemical type can only be either 'O' or 'C'!")
        else:
            print("Forcing chemical types as specified by force_chemtype...")
            fit.chemtype = ct

    return fit
