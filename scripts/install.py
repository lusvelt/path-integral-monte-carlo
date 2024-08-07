import subprocess
from .hooks import *
from .docs import *


def install():
    install_hooks()
    build_docs()
    subprocess.run(["pip", "install", "-e", "."])
