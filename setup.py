from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension


extensions = [Extension('cconf_gen',
                        ['autode/conformers/cconf_gen.pyx'])]

setup(
    name='autode',
    version='v1.0.0-alpha',
    packages=['autode', 'autode.conformers', 'autode.wrappers'],
    include_package_data=True,
    package_data={'': ['lib/Addition/*.obj', 'lib/Dissociation/*.obj', 'lib/Elimination/*.obj',
                       'lib/Rearrangement/*.obj', 'lib/Substitution/*.obj']},
    ext_modules=cythonize(extensions, language_level="3"),
    url='https://github.com/duartegroup/autodE',
    license='MIT',
    author='Tom Young',
    author_email='tom.young@chem.ox.ac.uk',
    description='Automated Transition State Finding'
)
