import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='blinkit',
    version='0.0.1',
    author='Vishnu Sangli',
    author_email='vsangli@berkeley.edu',
    description='Package for segemnting, characterizing, and clustering EOG blinks',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    project_urls = {
        "Plagiarism": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    },
    license='MIT',
    packages=['blinkit', 'blinkit.test'],
    #package_data={'blinkit': ['data/']}
)