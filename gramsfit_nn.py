import os
import urllib
from astropy.io import fits
from astropy.table import Table
import torch
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt


def download_grid():
    """
    STOP! Instead of calling this function, use gramsfit_utils.makegrid.
    see test_nn/test_nn.py for an example.
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


def do_NN(X, y, do_CV=False, best_model_file='best_model.pkl'):
    """
    Arguments:
    X: input features (grid parameters)
    y: target (logarithm of flux density stacked with the dependent parameters)
    best_model_file: name of file to save the best model to.
    do_CV: boolean, whether to perform cross-validation for NN hyperparameters
        CAUTION: this takes a long time!
        CV returns the best model and saves it to `best_model_file`.
        If `best_model_file` exists, it is read in; CV is not performed.
    Returns:
    best_model: the best model either from CV or from default hyperparameters.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # pipeline = Pipeline([('scaler', StandardScaler()),
    pipeline = Pipeline([('scaler', MinMaxScaler()),
                         ('model', MLPRegressor(max_iter=2000))
                         ])
    if do_CV:
        print("Performing cross-validation for NN hyperparameters.")
        if not os.path.isfile(best_model_file):
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
            joblib.dump(best_model, best_model_file)
        else:
            print("Using available best hyperparameter combination.")
            best_model = joblib.load(best_model_file)
    else:
        print("Not performing cross-validation for NN hyperparameters.")
        # The values below are based on inspecting the best model from CV.
        pipeline['model'].hidden_layer_sizes = (50, 50, 50)
        pipeline['model'].activation = 'logistic'
        pipeline['model'].alpha = 0.0001
        pipeline['model'].learning_rate_init = 0.001
        pipeline.fit(X_train, y_train)
        best_model = pipeline

    y_pred = best_model.predict(X_test)
    rel_err = np.nanmean(((y_test - y_pred) / y_pred)**2, axis=1)
    print('Relative error in fit to test sample: ')
    print('Median: {}%'.format(np.round(np.nanmedian(rel_err) * 100,
                                        decimals=2)))
    print('16th and 84th percentile: {}% and {}%'.format(
        np.round(np.nanpercentile(rel_err, 16) * 100, decimals=2),
        np.round(np.nanpercentile(rel_err, 84) * 100, decimals=2)))

    return best_model


def log_transform(X, cols=None, inverse=False):
    """
    Depending on the dynamic range of each feature, either
        return the untransformed feature or its logarithm.
    Arguments:
    X -- ndarray of input features
    cols -- array of booleans, indicating which columns to transform
            if None, cols are determined automatically
    inverse -- boolean, whether to invert the transformation
            in this case, cols must be specified
    Returns:
    X1 -- ndarray of transformed features
    cols -- array of booleans, indicating which columns were transformed
    """
    X1 = torch.empty_like(X)
    _ = X1.copy_(X)
    if inverse:
        # cols must have already been populated!
        if cols is None:
            raise ValueError("cols required for inversion! Run fit_nn first.")
        else:
            for i in range(len(cols)):
                if cols[i]:
                    X1[:, i] = 10**X[:, i]
        return X1, cols

    if cols is None:
        cols = np.zeros(X.shape[1], dtype=bool)
        par_range = np.nanmax(X, axis=0) / np.nanmin(X, axis=0)
        for i in range(len(cols)):
            if par_range[i] >= 100:
                cols[i] = True
                X1[:, i] = torch.log10(X[:, i])
    else:
        for i in range(len(cols)):
            if cols[i]:
                X1[:, i] = torch.log10(X[:, i])
    return X1, cols


def predict_nn(pipeline, X_test, cols=None, cols_dep=None):
    """Predict the spectrum for a given set of parameters.

    Arguments:
    pipeline: the trained neural network
    X_test: input features (grid parameters)
    cols: array of booleans, indicating which columns to transform

    Returns:
    y_pred: the predicted spectrum
    """
    if (cols is None) or (cols_dep is None):
        mesg = "cols and cols_dep must be specified! Run fit_nn first."
        raise ValueError(mesg)
    else:
        X_test1, _ = log_transform(X_test, cols)
        y_pred = pipeline.predict(X_test1)
        shape_y = y_pred.shape
        spec_pred = 10**y_pred[:, :shape_y[1]-len(cols_dep)]
        X_dep_pred, _ = log_transform(
                torch.as_tensor(y_pred[:, shape_y[1]-len(cols_dep):]),
                cols_dep, inverse=True)
        # shape_X = X_test.shape
        # # X_dep_pred is contained in the last len(cols_dep) columns
        # # of y_pred, but they will first have to be inverse-transformed
        # X_dep_pred, _ = log_transform(
        #         torch.as_tensor(y_pred[:, shape_y[1]-len(cols_dep):]),
        #         cols_dep, inverse=True)
        # X_dep_pred[:, 0] = torch.as_tensor(y_pred[:, shape_X[1]-len(cols_dep)])

    return spec_pred, X_dep_pred


def fit_nn(fitgrid, par_cols=None, dep_par_cols=None, do_CV=False,
           best_model_file='best_model.pkl'):
    """Fit a neural network to the grid of spectra.

    Arguments:
    fitgrid: astropy table with the grid of parameters and spectra to fit
    par_cols: names of the columns with the independent parameters
    dep_par_cols: names of the columns with the dependent parameters
    do_CV: boolean, whether to perform cross-validation
        CAUTION: this takes a long time!
        CV returns the best model and saves it to `best_model_file`.
        If `best_model_file` exists, it is read in; CV is not performed.

    Returns:
    pipeline: the trained neural network
    cols: ndarray(bool) indicates which independent parameters were transformed
    cols_dep:
            ndarray(bool) indicates which dependent parameters were transformed
    """
    # Features are obtained from the input grid parameters.
    # A feature is logarithmic if the corresponding parameter
    #   has a range greater than 100.
    if (par_cols is None) or (dep_par_cols is None):
        raise ValueError("par_cols and dep_par_cols must be specified!")
    else:
        X_in = torch.tensor(fitgrid[par_cols])
        X_dep_in = torch.tensor(fitgrid[dep_par_cols])
    X, cols = log_transform(X_in)
    X_dep, cols_dep = log_transform(X_dep_in)
    # The target is the logarithm of the flux density.
    y = torch.tensor(np.log10(fitgrid['Fspec']))
    y_all = torch.hstack((y, X_dep))

    return do_NN(X, y_all, best_model_file=best_model_file,
                 do_CV=do_CV), cols, cols_dep


def grid_fit_and_predict(fitgrid, predictgrid,
                         par_cols=['Lum', 'Teff', 'logg',
                                   'C2O', 'Rin', 'tau10'],
                         dep_par_cols=['Tin', 'DPR'],
                         best_model_file='best_model.pkl',
                         do_CV=False, return_best_model=False):
    """
    Arguments:
    fitgrid: astropy table with the grid of parameters and spectra to fit
    predictgrid: astropy table with the grid of parameters for which
        spectra are to be predicted
    par_cols: column names in fitgrid to use as parameters
    dep_par_cols: columns names in fitgrid that are derived from the
        base parameters in par_cols (namely, Tin, DPR)
    best_model_file: filename to save the best model to
    do_CV: boolean, whether to perform cross-validation for NN hyperparameters
        CAUTION: this takes a long time!
        CV returns the best model and saves it to best_model.pkl.
        If best_model.pkl exists, it is read in; CV is not performed.
    return_best_model: boolean, whether to return the best model

    Returns:
    The output of the NN is the spectrum as well as the derived parameters.

    predictgrid: for each input parameter combination, grid will now contain
        a column named 'Fspec_NN' with the predicted spectrum, and the
        dep_par_cols values will also be populated with the NN predictions.
    best_model: the best model either from CV or from default hyperparameters.
    """
    # # Features are obtained from the input grid parameters.
    # # A feature is logarithmic if the corresponding parameter
    # #   has a range greater than 100.

    # X = log_transform(torch.tensor(fitgrid[par_cols]), par_cols)
    # # The target is the logarithm of the flux density.
    # y = torch.tensor(np.log10(fitgrid['Fspec']))

    # # Train the neural network.
    # pipeline = do_NN(X, y, do_CV=do_CV)

    pipeline, cols, cols_dep = fit_nn(fitgrid, par_cols=par_cols,
                                      dep_par_cols=dep_par_cols, do_CV=do_CV,
                                      best_model_file=best_model_file)

    # Test neural network by passing only the relevant columns in an ndarray.
    X_test_in = torch.tensor(predictgrid[par_cols])
    X_test, _ = log_transform(X_test_in, cols)
    y_pred = pipeline.predict(X_test)
    shape_y = y_pred.shape
    spec_pred = 10**y_pred[:, :shape_y[1]-len(dep_par_cols)]
    X_dep_pred, _ = log_transform(
            torch.as_tensor(y_pred[:, shape_y[1]-len(dep_par_cols):]),
            cols_dep, inverse=True)
    # copy wavelength grid from input grid
    nlspec = fitgrid['Lspec'].shape[1]
    predictgrid['Lspec'] = np.zeros((len(predictgrid), nlspec))
    predictgrid['Lspec'] = fitgrid['Lspec'][0].flatten()
    predictgrid['Fspec_NN'] = spec_pred
    for i, col in enumerate(dep_par_cols):
        predictgrid[col] = np.array(X_dep_pred[:, i])

    # For a random spectrum from the test set,
    #   compare the prediction with the actual spectrum.
    if 'Fspec' in predictgrid.colnames:
        print("Comparing predicted and actual values for a random spectrum,")
        print("          the plot is saved to NN_fit.png.")
        j = np.random.choice(len(predictgrid), 1)[0]
        rel_err = ((predictgrid['Fspec'][j] - spec_pred[j, :]) /
                   predictgrid['Fspec'][j])**2
        print("Median relative error in predicting random spectrum: {}%".
              format(np.round(np.nanmedian(rel_err) * 100, decimals=2)))
        plt.plot(predictgrid['Lspec'][j], spec_pred[j, :], label='NN')
        plt.plot(predictgrid['Lspec'][j], predictgrid['Fspec'][j],
                 label='Grid')
        plt.xscale('log')
        plt.yscale('log')
        fmax = np.nanmax(predictgrid['Fspec'][j])
        plt.ylim(1e-5 * fmax, 1.05 * fmax)
        plt.legend(loc='best')
        plt.savefig('NN_fit.png', dpi=300, bbox_inches='tight')
    else:
        print("column 'Fspec' not in prediction grid, skipping plot.")

    if return_best_model:
        return cols, cols_dep, predictgrid, pipeline
    else:
        return cols, cols_dep
