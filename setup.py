from distutils.core import setup
  

setup(name='plan_abstractions',
      version='1.0.0',
      install_requires=[
            'hydra-core==1.0.4',
            'torch==1.9.0',
            'plotly',
            'wandb',
            #'isaacgym',
            #'isaacgym-utils',
            'pyquaternion',
            'numpy-quaternion',
            'pytorch_lightning==1.4.2',
            'shapely',
            'async-savers',
            'graphviz',
            'filelock',
            'cachetools',
            'seaborn',
            #'pybullet',
            'transformations',
            'gpytorch',
            'botorch',
            'matplotlib>=3.4'
      ],
      description='Planning using abstracted skill models',
      author='Alex LaGrassa, Jacky Liang, Saumya Saxena, Mohit Sharma, Shivam Vats',
      author_email='lagrassa@cmu.edu',
      url='none',
      packages=['plan_abstractions']
     )

