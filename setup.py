from setuptools import setup, find_packages

packages_ = find_packages()
packages = [p for p in packages_ if not(p == 'tests')]

setup(name='agents_option',
      version='0.T',
      description='A package to create agent following a Hierarchical Reinforcement Learning strategy',
      url='git@github.com:DamienAllonsius/agent_option_library.git',
      author='Damien Allonsius',
      author_email='allonsius.damien@hotmail.fr',
      license='MIT',
      packages=packages,
      install_requires=['numpy==1.15.4',
                        'setuptools==39.0.1',
                        'docopt==0.6.2',
                        'tqdm==4.29.0',
                        'python_dateutil==2.8.0',
                        'testresources==2.0.1 '],
      zip_safe=False)
