from setuptools import setup , find_packages

setup(
    name='Tuna_Recommender',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'torch',
    ],
    description='A Python library for creating neural recommender systems',
    author='tunaemirkandemir',
    author_email='tunakandemir@gmail.com',
    url='https://github.com/tunaemirkandemir/Tuna_Recommender_Library.git',  
)