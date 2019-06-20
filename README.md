# gramsfit
Code to fit GRAMS models to AGB/RSG candidates

The code requires numpy, scipy, and astropy installed.
Given a FITS file containing the source photometry and some related information (a band map into the GRAMS_filters.fits file, an array of True/False flags to denote whether a particular band is to be fit, another array of True/False flags denoting whether or not the source is detected in a given band, and the distance to the source in kpc), and the O-rich and C-rich model grid, find the best-fit models for each source.

How to use:

from gramsfit import *
ogrid = Table.read('grams_o.fits', format = 'fits')
cgrid = Table.read('grams_c.fits', format = 'fits')
data = Table.read('data.fits', format = 'fits')
#Choose 10 random sources from the data file (this speeds up the fit)
k = np.random.choice(len(data), size = 10)
data = data[k]
#The data file above has the ID, FITFLAG, and DKPC columns, so they're set to None below. Also, set scale to False for now.
fit = gramsfit(data, ogrid, cgrid, ID = None, FITFLAG = None, DKPC = None, scale = False)
