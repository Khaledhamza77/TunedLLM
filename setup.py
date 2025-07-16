import os
import setuptools

def read_requirements():
    try:
        with open(os.path.dirname(__file__) + '/requirements.txt', 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        print("requirements.txt not found. Proceeding without it.")
        return []

setuptools.setup(
    name='tunedLLM',
    version='0.0.1',
    author='Khaled Ibrahim',
    author_email='khaledhamza@aucegypt.edu',
    description='Agentic AI system for making your LLM tuned with latest research papers relevant to any query',
    url='https://github.com/Khaledhamza77/TunedLLM',
    packages=setuptools.find_packages(),
    install_requires=read_requirements(),
    python_requires='>=3.8'
)