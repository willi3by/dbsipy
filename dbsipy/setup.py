from setuptools import setup, find_packages

setup(
    name='dbsi',
    version='0.1',
    packages=find_packages(include=["dbsi", "dbsi.*"]),
    url='',
    license='Apache 2.0',
    author='Brady Williamson',
    author_email='brady.williamson@uc.edu',
    description='Commands for fitting DBSI data',
    install_requires=[
        'antspyx',
        'numpy',
        'scipy',
        'scikit-learn'
    ]
)
