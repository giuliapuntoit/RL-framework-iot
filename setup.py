import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="RL-framework-iot-giuliapuntoit",
    version="0.1.0",
    author="Giulia Milan",
    author_email="giuliapuntoit96@gmail.com",
    description="Reinforcement Learning algorithms for IoT Yeelight communication",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/giuliapuntoit/RL-framework-iot",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
