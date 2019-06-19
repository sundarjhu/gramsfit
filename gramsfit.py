from astropy.table import Table
from astropy.io import ascii
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

#def chisq2d(s, fdata, dfdata, uplim, fmod):
def chisq2d(s, data, fmod):
    """return chisq for one source and an entire model grid and 
    a vector of scale factors, one for each model"""
    nmodels = fmod.shape[0]
    nbands = len(np.where(data['FITFLAG'])[0])
    #model and data fluxes must have the same shape.
    #   also, the data fluxes are restricted using FITFLAG
    fm = fmod[:, data['BANDMAP'][data['FITFLAG']]]
    f = np.repeat(data['FLUX'][data['FITFLAG']][np.newaxis, :], nmodels, axis = 0)
    df = np.repeat(data['DFLUX'][data['FITFLAG']][np.newaxis, :], nmodels, axis = 0)
    #is the scale factor a scalar or a vector?
    if hasattr(s, '__len__'):
        ss = np.repeat(s[:, np.newaxis], nbands, axis = 1)
    else:
        ss = np.tile(s, (nmodels, nbands))
    #
    #The following chisq has two terms -- the first one for detections,
    #   the second for non-detections.
    #Two differences between the following and Sawicki et al. 2012, especially Eq. A10.
    #   1) He omitted one sigma_j in the proportionality in Eq. A6, which cancels
    #       when the substitution is made for the error function.
    #   2) I've used norm.cdf instead of the error function.
    residue = (f - ss * fm) / df
    chisq = np.nansum(residue[:, data['DETFLAG'][data['FITFLAG']]]**2, axis = 1) - 2 * \
        np.nansum(np.log(norm.cdf(residue[:, ~data['DETFLAG'][data['FITFLAG']]])), axis = 1)
    return chisq

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
    f = np.repeat(data['FLUX'][data['FITFLAG']][np.newaxis, :], n_models, axis = 0)
    df = np.repeat(data['DFLUX'][data['FITFLAG']][np.newaxis, :], n_models, axis = 0)
    fm = fmod[:, data['BANDMAP'][data['FITFLAG']]]

    #First, obtain the scale factor ignoring any non-detections.
    scale_det = np.nansum(f[:, data['DETFLAG'][data['FITFLAG']]] * \
                      fm[:, data['DETFLAG'][data['FITFLAG']]] / \
                      df[:, data['DETFLAG'][data['FITFLAG']]]**2, axis = 1) / \
        np.nansum(fm[:, data['DETFLAG'][data['FITFLAG']]]**2 / \
                  df[:, data['DETFLAG'][data['FITFLAG']]]**2, axis = 1)
    scale_nondet = scale_det.copy()

    #Only required if there are non-detections in the data.
    if (~data['DETFLAG'][data['FITFLAG']]).sum() != 0:
        for i in range(n_models):
            p = minimize(chisq1d, np.array([scale_det[i]]), args = (data, fmod[i, :]), \
                         method = 'Nelder-Mead', options = {'maxiter': 10000})
            if p['status'] == 0:
                scale_nondet[i] = p['x'][0]
            else:
                print("Search for scale_nondet did not converge for object " + \
                      "{}.".format(data['ID']))

    return scale_nondet

def get_chisq(data, ofmod, cfmod, scale = False):
    """Given the observed photometry and the model grid synthetic photometry,
    compute and output the chisq with and without the scale as a free parameter."""
    ndata = len(data)
    scale_o = np.tile(1.0, (ndata, ofmod.shape[0]))
    scale_c = np.tile(1.0, (ndata, cfmod.shape[0]))
    chisq_o = np.tile(np.nan, (ndata, ofmod.shape[0]))
    chisq_c = np.tile(np.nan, (ndata, cfmod.shape[0]))
    #Number of free parameters increased by one if scale is provided.
    p = 0
    if scale:
        for i in range(ndata):
            #number of bands that have finite fluxes AND have FITFLAG = True
            nfinite = len(np.where(~np.isnan(data[i]['FLUX'][data[i]['FITFLAG']]))[0])
            scale_o[i, :] = get_scale(data[i], ofmod)
            scale_c[i, :] = get_scale(data[i], cfmod)
            p = 1
            chisq_o[i, :] = chisq2d(scale_o[i, :], data[i], ofmod)/(nfinite - p)
            chisq_c[i, :] = chisq2d(scale_c[i, :], data[i], cfmod)/(nfinite - p)
    else:
        for i in range(ndata):
            #number of bands that have finite fluxes AND have FITFLAG = True
            nfinite = len(np.where(~np.isnan(data[i]['FLUX'][data[i]['FITFLAG']]))[0])
            chisq_o[i, :] = chisq2d(scale_o[i, :], data[i], ofmod)/(nfinite - p)
            chisq_c[i, :] = chisq2d(scale_c[i, :], data[i], cfmod)/(nfinite - p)
    #for each source, sort according to increase chisq. This sorted index into the model grid
    #   is stored in the modelindex arrays for each grid, and is output along with the sorted
    #   chisq values.
    modelindex_o = np.argsort(chisq_o, axis = 1)
    modelindex_c = np.argsort(chisq_c, axis = 1)
    chio = np.array([chisq_o[i, modelindex_o[i, :]] for i in range(ndata)])
    chisq_o = chio
    chic = np.array([chisq_c[i, modelindex_c[i, :]] for i in range(ndata)])
    chisq_c = chic
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

def par_summary(data, ogrid, cgrid, modelindex_o, modelindex_c, \
                scale_o, scale_c, flag_o, flag_c, chemtype, n_models = 100):
    """Boxplot summaries for the best-fit models of both chemistries to a single source.
    For each chemistry, the n_models models with the lowest chisq values are selected.
    For this to be true, the modelindex, scale, and flag arrays must already be sorted 
    in ascending order of chisq.
    data: 1-row astropy table. 
    modelindex_, scale_, flag_: 1 x ngrid numpy arrays where ngrid is 
        the number of models in that grid.
    chemtype: 1-element array ('o' or 'c') corresponding to given source.
    ogrid, cgrid: the full grid of models for both chemical types."""
    from ss_setPlotParams import setPlotParams
    plt = setPlotParams()
    plt.figure(figsize = (12, 3))

    def draw_plot(plt, data, edge_color, fill_color, data_labels = None):
        bp = plt.boxplot(data, labels = data_labels, patch_artist=True, \
                         meanline = True, showmeans = True)
        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color=edge_color)
        for patch in bp['boxes']:
            patch.set(facecolor=fill_color)
        return bp

    #draw the O-rich model boxplots
    flag = flag_o
    g = ogrid[modelindex_o[flag]][0:n_models]
    tau = g['tau10']
    taulabel = r"$\bm{-\log{\tau_{10}}}$"
    s = scale_o[flag][0:n_models]
    d = [np.log10(s * g['Lum']/1e3), np.log10(g['Teff']/1e3), np.log10(g['Rin']), \
         -np.log10(g['tau1']), -np.log10(tau), np.log10(np.sqrt(s) * g['MLR']/1e-13), \
         np.log10(g['Tin']/1e3), np.log10(s)]
    d_labels = [r"$\bm{\log{(L/10^3} \textbf{\textrm{L}}_\bm{\odot})}$", \
                r"$\bm{\log{(T_\textbf{\textrm{eff}}/10^3 \textbf{\textrm{K}})}}$", \
                r"$\bm{\log{(R_\textbf{\textrm{in}}/R_\textbf{\textrm{star}})}}$", \
                r"$\bm{-\log{\tau_1}}$", taulabel, \
                r"$\bm{\log{(\textbf{\textrm{DPR}}/10^{-13} \textbf{\textrm{M}}_\odot \textbf{\textrm{yr}}^{-1})}}$", \
                r"$\bm{\log{(T_\textbf{\textrm{in}}/10^3 \textbf{\textrm{K}})}}$", \
                r"$\bm{\log{s}}$"]
    bp = draw_plot(plt, d, 'blue', 'None', data_labels = d_labels)
    #draw the C-rich model boxplots
    flag = flag_c
    g = cgrid[modelindex_c[flag]][0:n_models]
    tau = g['tau11_3']
    taulabel = r"$\bm{-\log{\tau_{11.3}}}$"
    s = scale_c[flag][0:n_models]
    d = [np.log10(s * g['Lum']/1e3), np.log10(g['Teff']/1e3), np.log10(g['Rin']), \
         -np.log10(g['tau1']), -np.log10(tau), np.log10(np.sqrt(s) * g['MLR']/1e-13), \
         np.log10(g['Tin']/1e3), np.log10(s)]
    d_labels = [r"$\bm{\log{(L/10^3} \textbf{\textrm{L}}_\bm{\odot})}$", \
                r"$\bm{\log{(T_\textbf{\textrm{eff}}/10^3 \textbf{\textrm{K}})}}$", \
                r"$\bm{\log{(R_\textbf{\textrm{in}}/R_\textbf{\textrm{star}})}}$", \
                r"$\bm{-\log{\tau_1}}$", taulabel, \
                r"$\bm{\log{(\textbf{\textrm{DPR}}/10^{-13} \textbf{\textrm{M}}_\odot \textbf{\textrm{yr}}^{-1})}}$", \
                r"$\bm{\log{(T_\textbf{\textrm{in}}/10^3 \textbf{\textrm{K}})}}$", \
                r"$\bm{\log{s}}$"]
    bp = draw_plot(plt, d, 'red', 'None', data_labels = d_labels)
    #
    plt.xticks(fontsize = 8)
    plt.title("gramsfit parameter summary ({} best-fit models) for source {} (chemtype = {})".\
              format(n_models, data['ID'], chemtype), fontsize = 12)
    plt.show(block = False)

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

def gramsfit(data, ogrid, cgrid, ID = None, FITFLAG = None, DKPC = None, scale = False):
    #these inputs can either be a string pointing to the full path of the file,
    #   or an astropy table
    if isinstance(data, str):
        d = Table.read(data, format = 'fits')
        data = d
        d = 0.
    if isinstance(ogrid, str):
        d = Table.read(ogrid, format = 'fits')
        ogrid = d
        d = 0.
    if isinstance(cgrid, str):
        d = Table.read(cgrid, format = 'fits')
        cgrid = d
        d = 0.
    #
    ndata = len(data)
    if 'ID' not in data.columns:
        data['ID'] = np.arange(len(data))
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
    distscale = (np.repeat(data['DKPC'][:, np.newaxis], data['FLUX'].shape[1], axis = 1)/50.12)**2
    data['FLUX'] *= distscale
    data['DFLUX'] *= distscale

    if scale:
        chisq_o, chisq_c, modelindex_o, modelindex_c, scale_o, scale_c = \
            get_chisq(data, ogrid['Fphot'], cgrid['Fphot'], scale = True)
    else:
        chisq_o, chisq_c, modelindex_o, modelindex_c, scale_o, scale_c = \
            get_chisq(data, ogrid['Fphot'], cgrid['Fphot'])
    chemtype = get_chemtype(chisq_o, chisq_c)
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
    po, po_err = get_pars(ogrid, chisq_o, modelindex_o, scale_o, flag_o, n_accept = 100)
    pc, pc_err = get_pars(cgrid, chisq_c, modelindex_c, scale_c, flag_c, n_accept = 100)

    fit = Table([data['ID'], chemtype, chisq_o, chisq_c, modelindex_o, modelindex_c, \
                 scale_o, scale_c, flag_o, flag_c, po, po_err, pc, pc_err], \
                names = ('ID', 'chemtype', 'chisq_o', 'chisq_c', 'modelindex_o', \
                         'modelindex_c', 'scale_o', 'scale_c', 'flag_o', 'flag_c', \
                         'pars_o', 'pars_o_err', 'pars_c', 'pars_c_err'))

    #return chemtype, chisq_o, chisq_c, modelindex_o, modelindex_c, scale_o, scale_c, \
    #    flag_o, flag_c, po, po_err, pc, pc_err
    return fit
