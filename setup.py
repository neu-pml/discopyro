from setuptools import setup, find_packages
setup(
    name="discopyro-bpl",
    version="0.1",
    packages=find_packages(),

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=["docutils>=0.3"],

    # metadata to display on PyPI
    author="Eli Sennesh",
    author_email="sennesh.e@northeastern.edu",
    description="A generative model of compositionality in symmetric monoidal (Kleisli) categories",
    keywords="Bayesian program learning deep generative model compositionality category",
    project_urls={
        "Source Code": "https://github.com/neu-pml/discopyro",
    },
)
