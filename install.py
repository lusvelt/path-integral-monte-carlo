import os
import subprocess
from scripts import *

os.system('conda env create --file environment.yml')
os.system('conda init')
os.system('conda activate lqfn')

install_hooks()
build_docs()
subprocess.run(['pip', 'install', '-e', '.'])
