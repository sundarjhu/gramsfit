# gramsfit
Code to fit GRAMS models to AGB/RSG candidates

Requires the numpy, scipy, astropy, wget, and h5py packages. It also uses the pyphot fork available from https://github.com/sundarjhu/pyphot.

Given a FITS file containing the source photometry and some related information (an array of flags specifying whether a particular band is to be fit for a particular source, another array of flags that records non-detections, and an array with the distance in kpc to each source), and the O-rich and C-rich model grid, find the best-fit GRAMS models for each source.

Synthetic photometry for the GRAMS grid is computed using the pyphot package and by querying the Spanish Virtual Observatory's Filter Profile Service [http://svo2.cab.inta-csic.es/theory/fps/] for the filter response curves. For this to work, the user must provide a two-column comma-separated file named filters.csv. This file must contain a header row, although this information is not used. The first column can contain a user-defined name for each filter. The second column must contain the filter ID as specified on the SVO Filter Profile Service page (e.g., the 2MASS J filter is 2MASS/2MASS.J -- see http://svo2.cab.inta-csic.es/theory/fps/index.php?mode=browse&gname=2MASS)
The current version of the code does not accept user-defined filter response curves or filters that are not part of the SVO database.

How to use:
```
from gramsfit import *
#Download responses from the SVO Filter Profile Service for the filters specified in filters.csv
#    Save results in the HDF5 library filters.hd5, which will be used by pyphot
makeFilterSet(infile = 'filters.csv', libraryFile = 'filters.hd5')
#Generate synthetic photometry for the filters specified in filters.csv and filters.hd5
#    Results saved in grams_o.fits and grams_c.fits
makegrid(infile = 'filters.csv', libraryFile = 'filters.hd5')

ogrid = Table.read('grams_o.fits', format = 'fits') #could also pass filename as input instead
cgrid = Table.read('grams_c.fits', format = 'fits') #could also pass filename as input instead
data = Table.read('data.fits', format = 'fits') #could also pass filename as input instead
#Choose 10 random sources from the data file
k = np.random.choice(len(data), size = 10)
data = data[k]
data['FITFLAG'][:, 0:3] = False #(optional) Do not fit the first four wavelengths for all the sources
#The data file above has the ID, FITFLAG, and DKPC columns, so they're set to None below. Also, set scale to False for now.
#    Distance scaling hasn't been successfully implemented yet.
fit = gramsfit(data, ogrid, cgrid, ID = None, FITFLAG = None, DKPC = None, scale = False)
```
