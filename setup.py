from setuptools import setup , find_packages

setup(
    name='Tuna_Recommender',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        # add other dependencies here
    ],
    description='A Python library for creating neural recommender systems',
    author='Tuna Emir Kandemir',
    author_email='tunakandemir@gmail.com',
    url='https://github.com/yourusername/recommender_systems',  # update with your URL
)