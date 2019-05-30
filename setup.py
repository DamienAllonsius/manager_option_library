from setuptools import setup, find_packages

packages_ = find_packages()
packages = [p for p in packages_ if not(p == 'tests')]

setup(name='agents_option',
      version='0',
      description='A package to create agent following a Hierarchical Reinforcement Learning strategy',
      url='git@github.com:DamienAllonsius/agent_option_library.git',
      author='Damien Allonsius',
      author_email='allonsius.damien@hotmail.fr',
      license='MIT',
      packages=packages,
      install_requires=[
        'python-dateutil', 'docopt==0.6.2', 'numpy==1.15.4', 'scipy==1.1.0', 'sphinx', 'gym'
      ],
      zip_safe=False)
