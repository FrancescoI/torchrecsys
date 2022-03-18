from setuptools import find_packages, setup

setup(
    name='torchrecsys',
    version='v0.2.0',
    packages=find_packages(),
    install_requires=['torch', 'pandas', 'scipy', 'numpy', 'sklearn'],
    license='MIT',
    description='A PyTorch implementation of several collaborative filters and sequence model for recommendation systems',
    author='Francesco Imbriglia',
    author_email='francesco.imbriglia01@gmail.com',
    url='https://github.com/francescoi/torchrecsys',
    download_url='https://github.com/FrancescoI/torchrecsys/archive/refs/tags/v0.2.0.tar.gz',
    keywords=['recommendation', 'recommender', 'collaborative', 'filtering', 'sequence', 'model', 'pytorch'],
    classifiers=['Development Status :: 3 - Alpha',
                 'License :: OSI Approved :: MIT License',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence']
)