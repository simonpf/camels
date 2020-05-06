from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='camels',  # Required
    version='0.0.2',  # Required
    description='Interface to the CAMELS dataset.',
    long_description=long_description,
    long_description_content_type='text/markdown',  # Optional (see note above)
    url='https://github.com/simonpf/camels',  # Optional
    author='Simon Pfreundschuh',  # Optional
    author_email='simon.pfreundschuh@chalmers.se',  # Optional
    install_requires=["appdirs", "numpy", "pandas"],
    packages=["camels"],
    python_requires='>=3.6',
    project_urls={  # Optional
        'Source': 'https://github.com/simonpf/camels/',
    })
