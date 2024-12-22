from setuptools import setup

setup(
    name='logicai',
    version='0.1.0',
    packages=['logicai'],
    install_requires=[
        'colorama',
        'pydantic',
        'openai',
        'python-socketio'
    ],
    # entry_points={
    #     'console_scripts': [
    #         'logicai=logicai.main:main',
    #     ],
    # },
)
