import setuptools

REQUIRED_PACKAGES = [
    'tensorflow >= 1.1',
    'numpy >= 1.13',
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="easy_tfrecords",
    url="https://github.com/pckosek/easy_tfrecords",
    version="0.0.1",
    description="Package to streamline reading and writing data to tfrecord files",
    long_description=long_description,
    long_description_content_type="text/markdown",    
    author="Paul Kosek",    
    author_email="pckosek@fcps.edu",
    
    packages=setuptools.find_packages(),
    include_package_data=True,

    # Dependent packages (distributions)
    install_requires=REQUIRED_PACKAGES,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ], 
    keywords='tensorflow tensor machine learning tfrecord', 

)