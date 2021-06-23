from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension

extensions = [Extension('cconf_gen', ['autode/conformers/cconf_gen.pyx']),
              Extension('ade_dihedrals',
                        sources=['autode/ext/ade_dihedrals.pyx'],
                        include_dirs=['autode/ext/include'],
                        language='c++',
                        extra_compile_args=["-std=c++11"],
                        extra_link_args=["-std=c++11"]),

              Extension('ade_rb_opt',
                        sources=['autode/ext/ade_rb_opt.pyx'],
                        include_dirs=['autode/ext/include'],
                        language='c++',
                        extra_compile_args=["-std=c++11"],
                        extra_link_args=["-std=c++11"])]

setup(name='autode',
      version='1.0.5',
      packages=['autode',
                'autode.conformers',
                'autode.pes',
                'autode.path',
                'autode.neb',
                'autode.reactions',
                'autode.smiles',
                'autode.species',
                'autode.wrappers',
                'autode.transition_states',
                'autode.log',
                'autode.solvent'],
      include_package_data=True,
      package_data={'autode.transition_states': ['lib/*.txt']},
      ext_modules=cythonize(extensions, language_level="3"),
      url='https://github.com/duartegroup/autodE',
      license='MIT',
      author='Tom Young',
      author_email='tom.young@chem.ox.ac.uk',
      description='Automated Reaction Profile Generation')
