from setuptools import Extension, setup, find_packages, dist
from codecs import open
from os import path

dist.Distribution().fetch_build_eggs(['numpy>=1.11.2'])
try:
    import numpy as np
except ImportError:
    exit('Please install numpy>=1.11.2 first.')

try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True

__version__ = '1.1.2'

here = path.abspath(path.dirname(__file__))
# Get the long description from README.md
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# get the dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]

ext = '.pyx' if USE_CYTHON else '.c'
cmdclass = {}

ext = '.pyx' if USE_CYTHON else '.c'

extensions = [
    Extension(
        name='daisy.model.similarities', 
        sources=['daisy/model/similarities' + ext], 
        include_dirs=[np.get_include()]
    ),
    Extension(
        name='daisy.model.slim', 
        sources=['daisy/model/slim' + ext], 
        include_dirs=[np.get_include()]
    ),
    Extension(
        name='daisy.model.matrix_factorization', 
        sources=['daisy/model/matrix_factorization' + ext], 
        include_dirs=[np.get_include()]
    ),
    Extension(
        name='daisy.model.simlib_cython', 
        sources=['daisy/model/simlib_cython' + ext], 
        include_dirs=[np.get_include()]
    ),
]

if USE_CYTHON:
    ext_modules = cythonize(extensions)
    cmdclass.update({'build_ext': build_ext})
else:
    ext_modules = extensions

setup(
    name='scikit-daisy',
    version=__version__,
    author='',
    author_email='',
    maintainer='amazingDD',
    maintainer_email='',
    license='Apache License',
    packages=['daisy'],
    platforms=['all'],
    description=('An item-ranking library for recommender systems.'),
    # long_description=long_description,
    long_description='pytorch >= 1.0.0 is necessary',
    long_description_content_type='text/markdown',
    url='https://github.com/AmazingDD/daisyRec',
    keywords='recommender recommendation system item-rank',
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=install_requires,

    classifiers=[
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: Implementation',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries'
    ]
)

# how to send a package
# 1. python setup.py sdist build / python setup.py bdist_wheel --universal
# 2. pip install twine
# 3. twine upload dist/*