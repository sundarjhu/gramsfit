# import gramsfit
import gramsfit_utils
import gramsfit_nn
import pyphot as pyp
from astropy.table import Table
from astropy import units
import numpy as np
import os


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
og = Table.read('grams_o.fits', format='fits')
print('Column names in grid file: ', og.colnames)

"""
Synthetic photometry for an ndarray of spectra
"""
# Pick 100 random models from the above grid,
#   collect the spectra and compute photometry for this
#   "spectrum matrix"
k = np.random.choice(len(og), 100)
lspec = og['Lspec'][0] * units.um
# fspec is a ndarray of shape (nsources, len(lspec))
fspec = og['Fspec'][k, :] * units.Jy
filters_used = Table.read('filters.csv', format='csv',
                          names=('column', 'filterName'))
filterLibrary = pyp.get_library(fname='filters.hd5')
filterNames = [f['filterName'].replace('/', '_') for f in filters_used]
# seds is a ndarray of shape (nsources, nfilters)
_, seds = gramsfit_utils.synthphot(lspec, fspec, filterLibrary, filterNames)

"""
Neural network usage
Note: GridSearchCV is not performed. If `best_model.pkl` exists,
      the existing GridSearchCV result is used. If no such
      file exists, the default hyperparameters are set (based
      on an initial GridSearchCV run).
"""
fitgrid = og
# Example 1: select a subset of the grid as the prediction grid
predictgrid1 = og[np.random.choice(len(og), 5)].copy()
if os.path.exists('best_model.pkl'):
    gramsfit_nn.grid_fit_and_predict(fitgrid, predictgrid1, do_CV=True)
else:
    gramsfit_nn.grid_fit_and_predict(fitgrid, predictgrid1, do_CV=False)
# predictgrid1 now has a column called "Fspec_NN" with the predicted
#   spectra
print(predictgrid1['Fspec_NN'])

# Example 2: set the parameters of predictgrid
#   using an ndarray of shape (nsources, npars), with npars=7
#   with the following column names:
par_cols = ['Teff', 'logg', 'Mass', 'Rin', 'Tin',
            'tau10', 'Lum']
X_test = np.array([[3500, .5, 1.0, 2.1, 1300, 1.1, 1e4],
                   [4100, .5, 1.0, 2.3, 1100, 0.1, 8e3]])
predictgrid2 = Table(X_test, names=par_cols)
# Note: GridSearchCV is not performed. If `best_model.pkl` exists,
#       the existing GridSearchCV result is used.
if os.path.exists('best_model.pkl'):
    gramsfit_nn.grid_fit_and_predict(fitgrid, predictgrid2, do_CV=True)
else:
    gramsfit_nn.grid_fit_and_predict(fitgrid, predictgrid2, do_CV=False)
# predictgrid2 now has a column called "Fspec_NN" with the predicted
#   spectra
print(predictgrid2['Fspec_NN'])

