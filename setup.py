from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension


extensions = [Extension('cconf_gen', ['autode/conformers/cconf_gen.pyx'])]

setup(name='autode',
      version='1.0.0a2',
      packages=['autode',
                'autode.conformers',
                'autode.pes',
                'autode.neb',
                'autode.reactions',
                'autode.smiles',
                'autode.species',
                'autode.wrappers',
                'autode.transition_states',
                'autode.solvent'],
      include_package_data=True,
      package_data={'autode.transition_states': ['lib/*.txt']},
      ext_modules=cythonize(extensions, language_level="3"),
      url='https://github.com/duartegroup/autodE',
      license='MIT',
      author='Tom Young',
      author_email='tom.young@chem.ox.ac.uk',
      description='Automated Reaction Profile Generation')
