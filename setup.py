from os import path
from setuptools import setup, find_packages

classifiers = ["License :: OSI Approved :: MIT License"]


long_description = 'daisyRec is a Python toolkit developed for benchmarking top-N recommendation task.' \
                   'The name DAISY stands for multi-Dimension fAirly comparIson for recommender SYstem.'

# here = path.abspath(path.dirname(__file__))
# with open(path.join(here, 'README.md'), encoding='utf-8') as f:
#     long_description = f.read()
# with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
#     all_reqs = f.read().split('\n')

# install_requires = [x.strip() for x in all_reqs if 'git+' not in x]

install_requires = [
    'torch>=1.1.0',
    'numpy>=1.18.0',
    'scikit-learn>=1.0.2',
    'scipy>=1.5.2',
    'pandas>=1.0.5',
    'tqdm>=4.48.2',
    'pyyaml>=5.1.0',
    'colorlog==4.7.2',
    'colorama==0.4.4',
]

setup_requires = []

extras_require = {
    'optuna': ['optuna>=2.10.0']
}

print('=============================================')
print([package for package in find_packages() if package.startswith('daisy')])
print('=============================================')

setup(
    name = 'daisyRec',
    packages = [package for package in find_packages() if package.startswith('daisy')],
    # packages = find_packages(exclude=['tests*']),
    version = 'v2.0.6',  # Ideally should be same as your GitHub release tag varsion
    description=('An easy-to-use library for recommender systems.'),
    long_description = long_description,
    # long_description_content_type="text/markdown",
    author = 'Yu Di',
    author_email = 'di.yu.2021@mitb.smu.edu.sg',
    url = 'https://github.com/AmazingDD/daisyRec',
    download_url = 'https://github.com/AmazingDD/daisyRec/archive/refs/tags/v2.0.6.tar.gz',
    keywords = ['ranking', 'recommendation'],
    include_package_data=True,
    install_requires=install_requires,
    setup_requires=setup_requires,
    extras_require=extras_require,
    zip_safe=False,
    classifiers = classifiers,
)

# python setup.py sdist upload -r pypi