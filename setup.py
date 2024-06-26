from setuptools import setup, find_packages

setup(
    name='ensembletools',
    version='0.3',
    packages=find_packages(include=['ensembletools', 'ensembletools.*']),
    url='https://github.com/cubecloud/ensembletools.git',
    license='Apache-2.0',
    author='cubecloud',
    author_email='zenroad60@gmail.com',
    description='ensemble tools to work with models and predictions'
)
