from setuptools import setup, find_packages

setup(
    name='elastictorch',
    version='0.1.0',
    description='A PyTorch implementation of ElasticNet',
    author='Michael Nagle',
    author_email='michael.nagle@libd.org',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'torch',
    ],
)