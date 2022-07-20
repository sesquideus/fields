import setuptools


setuptools.setup(
    name='physics-fields',
    version='0.0.1',
    description='',
    url='http://github.com/sesquideus/physics-fields',
    author='Martin "Kvík" Baláž',
    author_email='martin.balaz@fmph.uniba.sk',
    license='MIT',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
    ],
    zip_safe=False
)
