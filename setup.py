import setuptools


setuptools.setup(
    name='physics-fields',
    version='0.1.0',
    description='azily evaluated scalar and vector fields for numpy. Includes scalar and vector Zernike polynomials',
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
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    zip_safe=False,
    keywords='mathematics physics fields visualisation',
)
