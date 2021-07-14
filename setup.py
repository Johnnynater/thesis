from setuptools import setup, find_packages

REQUIRED = [
    "numpy>=1.19.5",
    "pandas>=1.2.4",
    "ptype>=0.2.12",
    "greenery>=3.3.1",
    "scikit-learn>=0.24.1",
    "flair>=0.8.0.post1",
    "scipy>=1.6.2",
    "nltk>=3.6.2",
    "geopy>=2.1.0",
    "word2number>=1.1",
    "dirty-cat==0.1.0",
    "category-encoders>=2.2.2"
]

setup(
    name='auto_string_cleaner',
    version='1.0',
    package_dir={'': 'auto_string_cleaner'},
    packages=find_packages('auto_string_cleaner', exclude=['pfsms', 'gbc']),
    url='https://github.com/ml-tue/automated-string-cleaning',
    license='',
    author='John van Lith',
    author_email='jlith1997@gmail.com',
    description='Automated cleaning and encoding of strings in tabular data',
    install_requires=REQUIRED
)
