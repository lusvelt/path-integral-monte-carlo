import os

os.system('conda env create --file environment.yml')
os.system('conda init')
os.system('conda activate test')
os.system('python scripts.py install')
