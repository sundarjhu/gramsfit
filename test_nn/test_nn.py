# import gramsfit
import gramsfit_utils
import gramsfit_nn
import pyphot as pyp
from astropy.table import Table
from astropy import units
import numpy as np
import os
import torch

likelihood = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))

def lnlike(thetas, y, yerr):
    """Evaluate the log-likelihood for a given set of parameters.
    
    Arguments:
    thetas: ndarray of shape (npar, ngrid)
        the parameters for which the log-likelihood is to be evaluated
    y: ndarray of shape (nwave)
        the observed spectrum
    yerr: ndarray of shape (nwave)
        the uncertainty in the observed spectrum
    
    Returns:
    loglike: ndarray of shape (ngrid)
        the log-likelihood for each set of parameters

    Note:
    The function uses detbands and ndetbands, arrays of indices into
        the y, yerr denoting bands with and without detections respectively.
    """

    # First we need to predict the spectrum for each set of parameters.
    # This is done using the neural network.

    if torch.any(torch.isnan(thetas)):
        raise ValueError("NaNs in thetas!")
    if torch.any(thetas[:, cols] < 0):
        print(thetas[:, cols])
        print(cols)
        raise ValueError("Negative values in thetas!")

    spectra = gramsfit_nn.predict_nn(best_model, thetas, cols) * units.Unit(flux_unit)

    # now we compute the synthetic photometry

    _, seds = gramsfit_utils.synthphot(lspec, spectra, filterLibrary, filterNames)

    residue = torch.as_tensor(((seds - y) / yerr)) #.flatten()
    #import pdb; pdb.set_trace()

    logprob = torch.nansum(likelihood.log_prob(residue[:, detbands]), dim=1)
    if len(ndetbands) > 0:
        logprob -= torch.nansum(torch.log(likelihood.cdf(residue[:, ndetbands])), dim=1)
    
    # chisq = np.nansum(residue[detbands]**2)
    # if len(ndetbands) > 0:
    #     chisq -= 2 * np.nansum(np.log(norm.cdf(residue[ndetbands])))

    return logprob

def lnprior(thetas):
    """Evaluate the log-prior for a given set of parameters.
    
    Arguments:
    thetas: ndarray of shape (npar, ngrid)
        the parameters for which the log-prior is to be evaluated
    
    Returns:
    logprior: ndarray of shape (ngrid)
        the log-prior for each set of parameters
    """

    # This function should return -np.inf for any set of parameters
    #   that are outside the prior range, and 0 otherwise.
    #  For now, we assume a uniform prior over the entire parameter range.
    # Later we will create something more complex, based on the density of models in the training set.

    lp = torch.zeros(thetas.shape[0])
    # import pdb; pdb.set_trace()
    lp[torch.any(thetas < thetas_min, dim=1)] = -torch.inf
    lp[torch.any(thetas > thetas_max, dim=1)] = -torch.inf
    # lp[thetas[-1, :] < thetas_min] = -np.inf
    # lp[thetas[-1, :] > thetas_max] = -np.inf

    return lp # -np.inf if np.any((thetas < thetas_min) | (thetas > thetas_max)) else 0

def lnprob(thetas, y, yerr):
    # if thetas is a 1-dimensional array (only one set of parameters)
    #   then reshape it to a 2-dimensional array of shape (1, nparameters)
    if len(thetas.shape) == 1:
        thetas1 = torch.Tensor(thetas.reshape((1, len(thetas))))
    else:
        thetas1 = torch.Tensor(thetas)

    prob = lnprior(thetas1)
    prob[prob == 0] += lnlike(thetas1[prob==0], y, yerr)

    return prob

def do_MCMC(y, yerr, nwalkers=100, nsteps=1000, nburn=100):
    import emcee

    # Set up the sampler.
    ndim = len(thetas_min)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(y, yerr), vectorize=True)

    print("Minimum parameter values: ", thetas_min)
    print("Maximum parameter values: ", thetas_max)

    # Initialize the walkers.
    p0 = np.asarray([
        np.array(thetas_min + ((thetas_max - thetas_min) * np.random.rand(ndim)))
        for _ in range(nwalkers)
    ])

    # import pdb; pdb.set_trace()
    # print(p0.shape)

    # Run the burn-in phase.
    print("Running burn-in phase...")
    p0, _, _ = sampler.run_mcmc(p0, nburn, skip_initial_state_check=True, progress=True)
    print("Autocorrelation time for burn in: ", sampler.get_autocorr_time(quiet=True))
    sampler.reset()

    # Run the production phase.
    print("Running production phase...")
    sampler.run_mcmc(p0, nsteps, skip_initial_state_check=True, progress=True)

    return sampler

"""
Set up the filter library, download the grid,
and generate the synthetic photometry for the library.
"""
# Given a file filters.csv containing SVO-compatible filter names,
#   download the filter profiles and create a pyphot-compatible library
#   file filters.hd5
if not os.path.exists('filters.hd5'):
    gramsfit_utils.makeFilterSet(infile='filters.csv', libraryFile='filters.hd5')

# The following, from gramsfit_utils.py, will erase any existing
#   files named grams_o.fits and grams_c.fits, will download the originals
#   from figshare, then compute synthetic photometry over the filter set
#   specified by filters.csv and filters.hd5
if not os.path.exists('grams_o.fits'):
    gramsfit_utils.makegrid(infile='filters.csv', libraryFile='filters.hd5')

"""
Load the grid
"""
orich_grid = Table.read('grams_o.fits', format='fits')
print('Column names in grid file: ', orich_grid.colnames)

"""
Synthetic photometry for an ndarray of spectra
"""
# Pick 100 random models from the above grid,
#   collect the spectra and compute photometry for this
#   "spectrum matrix"
k = np.random.choice(len(orich_grid), 100)
lspec = orich_grid['Lspec'][0] * units.um
# fspec is a ndarray of shape (nsources, len(lspec))
fspec = orich_grid['Fspec'][k, :] * units.Jy
filters_used = Table.read('filters.csv', format='csv',
                          names=('column', 'filterName'))
filterLibrary = pyp.get_library(fname='filters.hd5')
filterNames = [f['filterName'].replace('/', '_') for f in filters_used]
filters = filterLibrary.load_filters(filterNames, interp=True, lamb=lspec)
# need to automatically grab the unit for lpivot
lpivots = np.array([f.lpivot.magnitude for f in filters])
# seds is a ndarray of shape (nsources, nfilters)
_, seds = gramsfit_utils.synthphot(lspec, fspec, filterLibrary, filterNames)

"""
Neural network usage
Note: GridSearchCV is not performed. If `best_model.pkl` exists,
      the existing GridSearchCV result is used. If no such
      file exists, the default hyperparameters are set (based
      on an initial GridSearchCV run).
"""
fitgrid = orich_grid
# Example 1: select a subset of the grid as the prediction grid
predictgrid1 = orich_grid[np.random.choice(len(orich_grid), 5)].copy()
if os.path.exists('best_model.pkl'):
    cols, _, best_model = gramsfit_nn.grid_fit_and_predict(fitgrid, predictgrid1, do_CV=True, return_best_model=True)
else:
    cols , _, best_model = gramsfit_nn.grid_fit_and_predict(fitgrid, predictgrid1, do_CV=False, return_best_model=True)
# predictgrid1 now has a column called "Fspec_NN" with the predicted
#   spectra
# print(predictgrid1['Fspec_NN'])

# Example 2: set the parameters of predictgrid
#   using an ndarray of shape (nsources, npars), with npars=7
#   with the following column names:
par_cols = ['Teff', 'logg', 'Mass', 'Rin', 'Tin',
            'tau10', 'Lum']
X_test = np.array([[3500, -.5, 1.0, 2.1, 1300, 1.1, 1e4],
                   [4100, -.5, 1.0, 2.3, 1100, 0.1, 8e3]])
predictgrid2 = Table(X_test, names=par_cols)
# Note: GridSearchCV is not performed. If `best_model.pkl` exists,
#       the existing GridSearchCV result is used.
if os.path.exists('best_model.pkl'):
    cols, _, best_model = gramsfit_nn.grid_fit_and_predict(fitgrid, predictgrid2, do_CV=True, return_best_model=True)
else:
    cols, _, best_model = gramsfit_nn.grid_fit_and_predict(fitgrid, predictgrid2, do_CV=False, return_best_model=True)
# predictgrid2 now has a column called "Fspec_NN" with the predicted
#   spectra
# print(predictgrid2['Fspec_NN'])

# print("Columns: ", cols)


def prepdata(data, filterFile='filters.csv'):
    # First, read in all available filters
    filters_used = Table.read('filters.csv', format='csv',
                              names=('column', 'filterName'))
    filterNames = np.array([f['filterName'].replace('/', '_')
                            for f in filters_used])
    
    filters = filterLibrary.load_filters(filterNames, interp=True, lamb=lspec)
    # need to automatically grab the unit for lpivot
    lpivots = np.array([f.lpivot.magnitude for f in filters])
    # Select the correct set of filters in the order
    #   specified by the BANDMAP column
    #   this will duplicate filters if necessary
    filterNames = filterNames[data['BANDMAP'][0]]
    lpivots = lpivots[data['BANDMAP'][0]]
    # Of these, select only the ones which are to be fit
    onlyfit = np.nonzero(data['FITFLAG'][0])[0]
    filterNames = filterNames[onlyfit]
    lpivots = lpivots[onlyfit]
    # Record bands which are and are not upper/lower limits
    detbands = np.nonzero(data['DETFLAG'][0][onlyfit])[0]
    ndetbands = np.nonzero(~data['DETFLAG'][0][onlyfit])[0]

    # Pass only the photometry to be fit to the fitting routine
    y = np.array(data['FLUX'][0][onlyfit])
    yerr = np.array(data['DFLUX'][0][onlyfit])

    return filterNames, lpivots, y, yerr, detbands, ndetbands


"""
Special case: data and grid have the same set of filters,
    but (a) there are multiple observations for a given filter
    and/or (b) there are some bands with detection limits and/or
               that are not to be fit.
    (a) is resolved using the BANDMAP column in the data table,
    (b) is resolved using the DETFLAG and FITFLAG columns.
"""
data = Table.read('fitterinput.vot', format='votable')
filterNames, lpivots, y, yerr, detbands, ndetbands = prepdata(data, filterFile='filters.csv')

modeldist = float(fitgrid.meta['DISTKPC'])
datadist = data['DKPC'][0]
distscale = (datadist / modeldist)**2
y = y * distscale
yerr = yerr * distscale

# The following is important since the synthphot function
#   (called in lnlike above) requires the flux and wavelength units.
# The wavelength units are specified whenever the filterLibrary is
#   created, so we only need to worry about the flux units below.
if data['FLUX'].unit is None:
    flux_unit = 'Jy'
else:
    flux_unit = data['FLUX'].unit.to_string()

# We need to set some ranges to get us started
# thetas_min = np.min(fitgrid[par_cols], axis=0)
# thetas_max = np.max(fitgrid[par_cols], axis=0)
thetas_min = torch.as_tensor(np.array(fitgrid[par_cols].to_pandas().min(axis=0)))
thetas_max = torch.as_tensor(np.array(fitgrid[par_cols].to_pandas().max(axis=0)))

sampler = do_MCMC(y, yerr, nwalkers=1000, nsteps=100, nburn=1000)

plot_ranges = [(thetas_min[i], thetas_max[i]) for i in range(len(thetas_min))]

# quick diagnostics: autocorrelation time
print("Autocorrelation time: ", sampler.get_autocorr_time(quiet=True))


# make the triangle plot
corner_mask = [True] * len(par_cols)
corner_mask[par_cols.index('logg')] = False
import corner
fig = corner.corner(sampler.flatchain[:, corner_mask], labels=np.asarray(par_cols)[corner_mask], use_arviz=False)  #, range=plot_ranges)
fig.savefig('triangle.png')

# now print the median and 1-sigma errors
print("16th, 50th and 84th percentiles:")
for i, par in enumerate(par_cols):
    print(par, np.percentile(sampler.flatchain[:, i], [16, 50, 84]))
# now the maximum likelihood parameters
maxlike = sampler.flatchain[np.argmax(sampler.flatlnprobability)]
print("Maximum likelihood parameters: " )
for i, par in enumerate(par_cols):
    print(par, maxlike[i])

# now plot the best-fit model and posterior samples
import matplotlib.pyplot as plt
bestfit = sampler.flatchain[np.argmax(sampler.flatlnprobability)]
bestfit = torch.as_tensor(bestfit.reshape((1, len(bestfit))))
bestfit_seds = gramsfit_nn.predict_nn(best_model, bestfit, cols)
bestfit_seds = bestfit_seds * units.Unit(flux_unit)
_, bestfit_phot = gramsfit_utils.synthphot(lspec, bestfit_seds, filterLibrary, filterNames)

print("Distance scale factor: ", distscale)
fig, ax = plt.subplots()
ax.errorbar(lpivots, y/distscale, yerr=yerr/distscale, fmt='o', zorder=10)
# plt.plot(filterNames, bestfit_phot, 'r-')
ax.plot(fitgrid['Lspec'][0], bestfit_seds[0, :]/distscale, 'r-', zorder=2)
# generate samples from the posterior
samples = sampler.flatchain[np.random.choice(len(sampler.flatchain), 100)]
sample_seds = gramsfit_nn.predict_nn(best_model, torch.as_tensor(samples), cols)
sample_seds = sample_seds * units.Unit(flux_unit)
ax.plot(fitgrid['Lspec'][0], sample_seds.T/distscale, 'k-', alpha=0.1, zorder=1)
ax.set_ylim(1e-5*np.max(y/distscale), 3 * np.max(y/distscale))
ax.set_yscale('log')
ax.set_xscale('log')
fig.savefig('bestfit.png')