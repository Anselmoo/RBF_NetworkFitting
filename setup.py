# List all dependencies of RBF_NetworkFitting
# Requires pip >= 18.0

try:
	from setuptools import setup
except ImportError:
	from distutils.core import setup

with open('requirements.txt') as f:
	required = f.read().splitlines()

__authors__ = ['Anselm Hahn']
__license__ = 'MIT'
__version__ = '0.5'
__date__ = '16/09/2019'

setup(
		name='RRBF_NetworkFitting',
		python_requires='>=3.5.0',
		version=__version__,
		packages=['RBFN', 'test', 'examples'],
		url='https://github.com/Anselmoo/RRBF_NetworkFitting',
		license=__license__,
		author=__authors__,
		author_email='Anselm.Hahn@gmail.com',
		description='Radial basis function network with genetic optimizer',
		platforms=['MacOS :: MacOS X', 'Microsoft :: Windows',
		           'POSIX :: Linux'],
)
