import os
import urllib
from astropy.io import fits
from astropy.table import Table
import torch
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt


def download_grid():
    """
    STOP! Instead of calling this function, use gramsfit_utils.makegrid.
    see test_nn.py for an example.
    """
    file_link = {'o': 'https://ndownloader.figshare.com/files/9684331',
                 'c': 'https://ndownloader.figshare.com/files/9684328'}
    for c in ['o', 'c']:
        file_name = f'grams_{c}.fits'
        if not os.path.exists(file_name):
            print(f'Downloading {file_name}')
            urllib.request.urlretrieve(file_link[c], file_name)
            grid, header = fits.getdata(file_name, header=True)
            grid = Table(grid)
            print("Renaming 'MLR' column to 'DPR'")
            grid.rename_column('MLR', 'DPR')
            g = fits.table_to_hdu(grid)
            g.header = header
            g.writeto(file_name, overwrite=True)
        else:
            print(f'{file_name} already exists')


def do_NN(X, y, do_CV=False):
    """
    Arguments:
    X: input features (grid parameters)
    y: target (logarithm of flux density)
    do_CV: boolean, whether to perform cross-validation for NN hyperparameters
        CAUTION: this takes a long time!
        CV returns the best model and saves it to best_model.pkl.
        If best_model.pkl exists, it is read in; CV is not performed.
    Returns:
    best_model: the best model either from CV or from default hyperparameters.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    pipeline = Pipeline([('scaler', StandardScaler()),
                         ('model', MLPRegressor(max_iter=1000))
                         ])
    if do_CV:
        print("Performing cross-validation for NN hyperparameters.")
        if not os.path.isfile('best_model.pkl'):
            param_grid = {
                    'model__hidden_layer_sizes': [(100, 50,),
                                                  (50, 50, 50,),
                                                  (100, 100, 100,)],
                    'model__activation': ['relu', 'logistic', 'tanh'],
                    'model__alpha': [0.0001, 0.001, 0.01],
                    'model__learning_rate_init': [0.001, 0.01, 0.1]
                    }

            grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5)

            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            joblib.dump(best_model, 'best_model.pkl')
        else:
            print("Using available best hyperparameter combination.")
            best_model = joblib.load('best_model.pkl')
    else:
        print("Not performing cross-validation for NN hyperparameters.")
        pipeline['model'].hidden_layer_sizes = (100, 100, 100)
        pipeline['model'].activation = 'relu'
        pipeline['model'].alpha = 0.0001
        pipeline['model'].learning_rate_init = 0.001
        pipeline.fit(X_train, y_train)
        best_model = pipeline

    y_pred = best_model.predict(X_test)
    rel_err = np.nanmean(((y_test - y_pred) / y_pred)**2, axis=1)
    print('Median relative error in fit to test sample: ',
          np.nanmedian(rel_err))

    return best_model


def interp_grid(pipeline, X_test):
    return pipeline.predict(X_test)


def log_transform(X, par_cols):
    par_range = np.nanmax(X, axis=0) / np.nanmin(X, axis=0)
    for i in range(len(par_range)):
        if par_range[i] >= 100:
            X[:, i] = torch.log10(X[:, i])
    return X

def fit_nn(fitgrid, do_CV=False):
    """Fit a neural network to the grid of spectra.

    Arguments:
    fitgrid: astropy table with the grid of parameters and spectra to fit
    do_CV: boolean, whether to perform cross-validation
        CAUTION: this takes a long time!
        CV returns the best model and saves it to best_model.pkl.
        If best_model.pkl exists, it is read in; CV is not performed in this case.

    Returns:
    pipeline: the trained neural network
    """
    # Features are obtained from the input grid parameters.
    # A feature is logarithmic if the corresponding parameter
    #   has a range greater than 100.
    # par_cols = ['Teff', 'logg', 'Mass', 'C2O', 'Rin', 'tau1', 'tau10', 'Lum', 'DPR', 'Tin']
    # TBD: change this to only 7 parameters:
    par_cols = ['Teff', 'logg', 'Mass', 'Rin', 'Tin', 'tau10', 'Lum']
    # But this will require re-running the GridSearchCV.

    X = log_transform(torch.tensor(fitgrid[par_cols]), par_cols)
    # The target is the logarithm of the flux density.
    y = torch.tensor(np.log10(fitgrid['Fspec']))

    return do_NN(X, y, do_CV=do_CV)


def grid_fit_and_predict(fitgrid, predictgrid,
                         do_CV=False, return_best_model=False):
    """
    Arguments:
    fitgrid: astropy table with the grid of parameters and spectra to fit
    predictgrid: astropy table with the grid of parameters for which
        spectra are to be predicted
    do_CV: boolean, whether to perform cross-validation for NN hyperparameters
        CAUTION: this takes a long time!
        CV returns the best model and saves it to best_model.pkl.
        If best_model.pkl exists, it is read in; CV is not performed in this case.
    return_best_model: boolean, whether to return the best model

    Returns:
    predictgrid: the table will now have a column named 'Fspec_NN'.
    best_model: the best model either from CV or from default hyperparameters.
    """
    # # Features are obtained from the input grid parameters.
    # # A feature is logarithmic if the corresponding parameter
    # #   has a range greater than 100.
    # # par_cols = ['Teff', 'logg', 'Mass', 'C2O', 'Rin', 'tau1', 'tau10', 'Lum', 'DPR', 'Tin']
    # # TBD: change this to only 7 parameters:
    # par_cols = ['Teff', 'logg', 'Mass', 'Rin', 'Tin', 'tau10', 'Lum']
    # # But this will require re-running the GridSearchCV.

    # X = log_transform(torch.tensor(fitgrid[par_cols]), par_cols)
    # # The target is the logarithm of the flux density.
    # y = torch.tensor(np.log10(fitgrid['Fspec']))

    # # Train the neural network.
    # pipeline = do_NN(X, y, do_CV=do_CV)

    pipeline = fit_nn(fitgrid, do_CV=do_CV)

    # Test the neural network.
    X_test = log_transform(torch.tensor(predictgrid[par_cols]), par_cols)
    y_pred = interp_grid(pipeline, X_test)
    spec_pred = 10**y_pred
    predictgrid['Fspec_NN'] = spec_pred

    # For a random spectrum from the test set,
    #   compare the prediction with the actual spectrum.
    if 'Fspec' in predictgrid.colnames:
        print("Comparing the prediction and actual values for a random spectrum,")
        print("          the plot is saved to NN_fit.png.")
        j = np.random.choice(len(predictgrid), 1)[0]
        rel_err = ((predictgrid['Fspec'][j] - spec_pred[j, :]) /
                   predictgrid['Fspec'][j])**2
        print("Median relative error in predicting random spectrum: ",
              np.nanmedian(rel_err))
        plt.plot(predictgrid['Lspec'][j], spec_pred[j, :], label='NN')
        plt.plot(predictgrid['Lspec'][j], predictgrid['Fspec'][j], label='Grid')
        plt.xscale('log')
        plt.yscale('log')
        fmax = np.nanmax(predictgrid['Fspec'][j])
        plt.ylim(1e-5 * fmax, 1.05 * fmax)
        plt.legend(loc='best')
        plt.savefig('NN_fit.png', dpi=300, bbox_inches='tight')
    else:
        print("column 'Fspec' not available in prediction grid, skipping plot.")

    if return_best_model:
        return predictgrid, pipeline
