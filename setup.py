from setuptools import setup
from Cython.Build import cythonize


setup(
    name='autode',
    version='v1.0.0-alpha',
    packages=['autode'],
    include_package_data=True,
    package_data={'': ['lib/Addition/*.obj', 'lib/Dissociation/*.obj', 'lib/Elimination/*.obj',
                       'lib/Rearrangement/*.obj', 'lib/Substitution/*.obj']},
    ext_modules=cythonize('autode/conf_gen.pyx', language_level="3"),
    url='',
    license='MIT',
    author='Tom Young',
    author_email='tom.young@chem.ox.ac.uk',
    description='Automated Transition State Finding'
)
