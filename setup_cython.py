from distutils.core import setup
from Cython.Distutils import Extension
from Cython.Distutils import build_ext
import numpy as np
import cython_gsl

setup(
    name="stopsignal",
    version="0.1a",
    author="Thomas V. Wiecki, Dora Matzke, Eric-Jan Wagenmakers",
    author_email="thomas_wiecki@brown.edu",
    url="http://github.com/twiecki/stopsignal",
    packages=["stopsignal"],
    package_data={"stopsignal":["examples/*.csv"]},
    description="""stopsignal implements a hierarchical Bayesian estimation of the stopsignal model presented in Matzke et al (submitted) in kabuki.
    Matzke, D., Dolan, C.V, Logan, G.D., Brown, S.D., & Wagenmakers, E.-J. (submitted). Bayesian parametric estimation of stop-signal reaction time distributions. Manuscript submitted for publication.""",
    setup_requires=['NumPy >=1.3.0', 'kabuki >= 0.2a', 'pymc', 'cython_gsl'],
    include_dirs = [np.get_include(), cython_gsl.get_include()],
    cmdclass = {'build_ext': build_ext},
    classifiers=[
                'Development Status :: 3 - Alpha',
                'Environment :: Console',
                'Operating System :: OS Independent',
                'Intended Audience :: Science/Research',
                'License :: OSI Approved :: GNU General Public License (GPL)',
                'Programming Language :: Python',
                'Topic :: Scientific/Engineering',
                 ],
    ext_modules = [Extension("stop_likelihoods",
                             ["src/stop_likelihoods.pyx"],
                             libraries=['gsl','gslcblas'],
                             library_dirs=[cython_gsl.get_library_dir()],
                             cython_include_dirs=[cython_gsl.get_cython_include_dir()])]
)       
