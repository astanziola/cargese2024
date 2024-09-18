from setuptools import setup, find_packages

setup(
    name="jwavetutorial",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # List your project dependencies here
        # For example:
        # 'requests>=2.25.1',
    ],
    author="Antonio Stanziola",
    author_email="stanziola.antonio@gmail.com",
    description="jax and jwave tutorial for the European Summer School on Physical Acoustics and its Applications",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/astanziola/cargese2024",
    python_requires='>=3.10',
)