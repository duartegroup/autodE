from setuptools import setup

setup(
    name='autode',
    version='v1.0.0-alpha',
    packages=['autode'],
    include_package_data=True,
    package_data={'': ['lib/Addition/*.obj', 'lib/Dissociation/*.obj', 'lib/Elimination/*.obj',
                       'lib/Rearrangement/*.obj', 'lib/Substitution/*.obj']},
    url='',
    license='MIT',
    author='Tom Young',
    author_email='tom.young@chem.ox.ac.uk',
    description='Automated Transition State Finding'
)
