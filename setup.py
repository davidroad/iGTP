from setuptools import setup, find_packages

setup(
    name='iGTP',
    version='0.1',
    packages=find_packages(where='iGTP'),
    package_dir={'': 'iGTP'},
    install_requires=[
        'scanpy',
        'pyyaml',
        # add other dependencies here
    ],
    entry_points={
        'console_scripts': [
            'igtp-train=iGTP.iGTP_Kfold_train:main',
        ],
    },
)
