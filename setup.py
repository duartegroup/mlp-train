from setuptools import setup


setup(name='mltrain',
      version='1.0.0a1',
      description='Machine Learning Potential Training',
      packages=['mltrain',
                'mltrain.configurations',
                'mltrain.loss',
                'mltrain.training',
                'mltrain.potentials',
                'mltrain.potentials.gap',
                'mltrain.sampling',
                'mltrain.potentials.ace',
                'mltrain.potentials.nequip'],
      url='https://github.com/t-young31/mltrain',
      license='MIT',
      author='Tom Young, Tristan Johnston-Wood',
      author_email='tom.young@chem.ox.ac.uk')
