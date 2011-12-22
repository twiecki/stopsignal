from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
import os
import sys
print sys.platform

if sys.platform == "win32":
    gsl_include = r"c:\Program Files\GnuWin32\include"
    lib_gsl_dir = r"c:\Program Files\GnuWin32\lib"
else:
    gsl_include = os.popen('gsl-config --cflags').read()[2:-1]
    lib_gsl_dir = ''

if gsl_include == '':
    print "Couldn't find gsl-config. Make sure it's installed and in the path."
    sys.exit(-1)

setup(
    name="stopsignal",
    version="0.1a",
    author="Thomas V. Wiecki, Dora Matzke, Eric-Jan Wagenmakers",
    author_email="thomas_wiecki@brown.edu",
    url="http://github.com/twiecki/stopsignal",
    packages=["stopsignal"],
    package_data={"stopsignal":["examples/*.csv"]},
    description="""stopsignal implements a hierarchical Bayesian estimation of the stopsignal model presented in Matzke et al(2011) in kabuki.
    Matzke, D., Dolan, C.V, Logan, G.D., Brown, S.D., & Wagenmakers, E.-J. (2011). Bayesian parametric estimation of stop-signal reaction time distributions. Manuscript submitted for publication.""",
    install_requires=['NumPy >=1.3.0', 'kabuki >= 0.2a', 'pymc'],
    setup_requires=['NumPy >=1.3.0', 'kabuki >= 0.2a', 'pymc'],
    include_dirs = [np.get_include(), gsl_include],
    library_dirs=[lib_gsl_dir],
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
    ext_modules = [Extension("stop_likelihoods", ["src/stop_likelihoods.pyx"], libraries=['gsl','gslcblas'])]
)
