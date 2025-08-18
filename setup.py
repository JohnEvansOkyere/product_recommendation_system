from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

## edit below variables as per your requirements -
REPO_NAME = "Book-Recommendation-System"
AUTHOR_USER_NAME = "johnEvansOkyere"
SRC_REPO = "books_recommender"
LIST_OF_REQUIREMENTS = []


setup(
    name=SRC_REPO,
    version="0.0.1",
    author="John Evans Okyere",
    description="A small local packages for bOok recommendation system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JohnEvansOkyere/product_recommendation_system.git",
    author_email="okyerevansjohn@gmail.com",
    packages=find_packages(), # automatically find packages in the src directory, it looks for __init__.py files
    license="MIT",
    python_requires=">=3.7",
    install_requires=LIST_OF_REQUIREMENTS
)