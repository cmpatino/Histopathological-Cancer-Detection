from distutils.core import setup


def readme():
    """Import the README.md Markdown file and try to convert it to RST format."""
    try:
        import pypandoc
        return pypandoc.convert('README.md', 'rst')
    except(IOError, ImportError):
        with open('README.md') as readme_file:
            return readme_file.read()


setup(
    name='cancer_detection',
    version='0.1',
    description='Recognition of cancerous tissue using deep learning',
    long_description=readme(),
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    url='https://github.com/cmpatino/Histopathological-Cancer-Detection.git',
    author='Carlos Miguel Patiño',
    author_email='carlosmiguelpatino@gmail.com',
    license='...',
    packages=['cancer_detection'],
    install_requires=[
        'pypandoc>=1.4',
        'watermark>=1.8.1',
        'pandas>=0.24.2',
        'scikit-learn>=0.20.3',
        'scipy>=1.2.1',
        'matplotlib>=3.0.3',
        'pytest>=4.3.1',
        'pytest-runner>=4.4',
        'click>=7.0'
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)
