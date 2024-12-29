from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name='LLM_Merge',
    version='0.0.1',
    description='LLM_Merge',
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        'torch>=1.6.0',
        'transformers>=4.18.0',
        'datasets',
        'accelerate>=0.20.1'
    ],
)
