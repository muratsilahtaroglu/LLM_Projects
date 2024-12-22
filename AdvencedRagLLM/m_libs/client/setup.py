from setuptools import setup

setup(
    name='client',
    version='0.1.0',
    packages=['ollama_client'],
    install_requires=[
        'pandas',
        'ollama',
        'rouge_score',
        'sentence_transformers'
    ]
)
