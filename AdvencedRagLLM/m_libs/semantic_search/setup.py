from setuptools import setup

setup(
    name='semantic_search',
    version='0.1.0',
    packages=['semantic_search'],
    install_requires=[
        'pandas',
        'pydantic',
        'nltk',
        'fastapi'
    ]
)
