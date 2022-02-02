from setuptools import find_packages, setup


# Import version
#__builtins__.__SPOTLIGHT_SETUP__ = True

setup(
    name='recall',
    version='v0.1.0',
    packages=find_packages(),
    install_requires=['torch', 'pandas', 'scipy', 'numpy', 'sklearn'],
    license='MIT',
    classifiers=['Development Status :: 3 - Alpha',
                 'License :: OSI Approved :: MIT License',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence'],
)