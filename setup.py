from setuptools import setup

setup(name='xesmf',
      version='0.1.0',
      description='use ESMF regridding through xarray',
      url='http://github.com/jiaweizhuang/xesmf',
      author='Jiawei Zhuang',
      author_email='jiaweizhuang@g.harvard.edu',
      license='MIT',
      packages=['xesmf'],
      install_requires=['esmpy','xarray','numpy','scipy']
      )
