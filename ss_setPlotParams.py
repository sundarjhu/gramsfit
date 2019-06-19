import matplotlib.pyplot as plt

def setPlotParams():
    plt.figure(figsize = (8, 8))
    params = {'legend.fontsize': 'x-large',
              'axes.labelsize':20,
              'axes.titlesize':20,
              'xtick.labelsize':20,
              'ytick.labelsize':20,
              'text.usetex': True,
              'text.latex.preamble': r'\usepackage{bm}'}
    plt.rcParams.update(params)
    return plt
