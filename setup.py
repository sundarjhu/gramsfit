from setuptools import setup, find_packages

setup(
    name='gramsfit',
    version='1.0',
    packages=find_packages(),
    py_modules=[
        'gramsfit',
        'gramsfit_utils',
        'gramsfit_nn',
    ],
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'astropy',
        'torch',
        'pyphot',
        'h5py',
        'scikit-learn',
        'corner',
        'tqdm',
    ],
    # entry_points={
    #     'console_scripts': [
    #         'your-script-name=your_package_name.module_name:main',
    #     ],
    # }
)
