import setuptools

REQUIRED_PACKAGES = [
    'tensorflow >= 1.1',
    'numpy >= 1.13',
]

setuptools.setup(
    name="easy_tfrecords",
    url="https://github.com/pckosek/tfrecordutils",
    version="0.0.1",
    description="Package to streamline reading and writing data to tfrecord files",
    author="Paul Kosek",

    packages=["easy_tfrecords"],
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