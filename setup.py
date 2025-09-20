from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="floodwise_ph",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Streamlit application for exploring flood control projects in the Philippines",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/floodwise-ph",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    install_requires=[
        'streamlit>=1.24.0,<2.0.0',
        'pandas>=2.0.0,<3.0.0',
        'numpy>=1.24.0,<2.0.0',
        'python-dotenv>=1.0.0,<2.0.0',
        'openai>=1.0.0,<2.0.0',
        'fuzzywuzzy>=0.18.0,<1.0.0',
        'python-Levenshtein>=0.12.2,<1.0.0',
        'httpx>=0.24.0,<1.0.0',
    ],
    entry_points={
        'console_scripts': [
            'floodwise-ph=app:main',
        ],
    },
)
