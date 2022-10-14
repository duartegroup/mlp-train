from setuptools import setup


setup(name='mlptrain',
      version='1.0.0a2',
      description='Machine Learning Potential Training',
      packages=['mlptrain',
                'mlptrain.configurations',
                'mlptrain.loss',
                'mlptrain.training',
                'mlptrain.sampling',
                'mlptrain.potentials',
                'mlptrain.potentials.gap',
                'mlptrain.potentials.ace',
                'mlptrain.potentials.mace',
                'mlptrain.potentials.nequip'],
      url='https://github.com/t-young31/mlptrain',
      license='MIT',
      author='Tom Young, Tristan Johnston-Wood')
