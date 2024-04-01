import gramsfit_utils as gu
from astropy.table import Table
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


mesg = """
        This script demonstrates how to specify user-defined filters
            in which to compute GRAMS synthethic photometry. We will
            use a dummy filter centred around 15 micron as an example.
        Two files are provided with this script:
        - filters_dummy.csv: a csv file containing the filter information
        - dummy_filter.vot: a VOTable file containing a dummy
            filter transmission curve

        Inspect the contents of both files along with the documentation
        for gramsfit_utils.makeFilterSet() as an illustration of how
        to specify user-defined filters.



    """
print(mesg)
input("Press Enter to continue...")
gu.makeFilterSet(infile='filters_dummy.csv',
                 libraryFile='filters_dummy.hd5',
                 from_SVO=False)
gu.makegrid(infile='filters_dummy.csv',
            libraryFile='filters_dummy.hd5',
            outfile_suffix='_dummy')

cg = Table.read('grams_c_dummy.fits', format='fits')
plt.plot(cg['Lspec'][0], cg['Fspec'][0], label='GRAMS model spectrum')
plt.plot([15.0], cg['Fphot'][0], 'ro',
         label='Synthetic photometry in dummy filter')
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='best')
plt.xlabel('Wavelength (micron)')
plt.ylabel('Flux (Jy)')
plt.show()
