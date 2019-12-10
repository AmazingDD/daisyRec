'''
@Author: Yu Di
@Date: 2019-12-02 13:22:54
@LastEditors: Yudi
@LastEditTime: 2019-12-10 13:00:30
@Company: Cardinal Operation
@Email: yudi@shanshu.ai
@Description: 
'''
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

__version__ = '1.0.0'

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
]

if USE_CYTHON:
    ext_modules = cythonize(extensions)
    cmdclass.update({'build_ext': build_ext})
else:
    ext_modules = extensions

setup(
    name='daisy',
    author='Yu Di',
    author_email='yudi_mars@126.com',
    version=__version__,
    description=('An item-ranking library for recommender systems.'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='http://amazingdd.github.io',
    license='MIT',
    keywords='recommender recommendation system item-rank',

    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=install_requires,
)