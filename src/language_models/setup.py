from setuptools import setup, find_packages


setup(
   name='searchtool_ingestion',
   version='1.0',
   description="searchtool's ingestion package",
   author='Fraunhofer IAIS',
   packages=find_packages(),
   install_requires=['flask', 'requests', 'pdfminer', 'pdfminer.six', 'tqdm', 'yapp'],
)
