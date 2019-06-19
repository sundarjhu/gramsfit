# gramsfit
Code to fit GRAMS models to AGB/RSG candidates

The code requires numpy, scipy, and astropy installed.
Given a FITS file containing the source photometry and some related information (a band map into the GRAMS_filters.fits file, an array of True/False flags to denote whether a particular band is to be fit, another array of True/False flags denoting whether or not the source is detected in a given band, and the distance to the source in kpc), and the O-rich and C-rich model grid, find the best-fit models for each source.
