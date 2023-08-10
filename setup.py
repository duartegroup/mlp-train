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
                'mlptrain.potentials.nequip',
                'mlptrain.potentials.mace'],
      url='https://github.com/duartegroup/mlp-train',
      license='MIT',
      author='mlp-train authors')
