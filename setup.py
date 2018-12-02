import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="easy_tfrecords",
    url="https://github.com/pckosek/easy_tfrecords",
    version="0.1.0",
    description="Package to streamline reading and writing data to tfrecord files",
    long_description=long_description,
    long_description_content_type="text/markdown",    
    author="Paul Kosek",    
    author_email="pckosek@fcps.edu",
    
    packages=setuptools.find_packages(),
    include_package_data=False,

    # Dependent packages (distributions)
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ], 
    keywords='tensorflow tensor machine learning tfrecord', 

)