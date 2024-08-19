import os
import subprocess
from .docs import build_docs

root_dir = os.getcwd()
hooks_dir = os.path.join(root_dir, ".git", "hooks")


def install_hooks():
    subprocess.run(["pre-commit", "install"])
    post_update_path = os.path.join(hooks_dir, "post-update")
    with open(post_update_path, "w") as file:
        file.write("#!/bin/bash\npython scripts.py post-update-hook\n")
        file.close()
    file.close()
    os.chmod(post_update_path, 0o755)
    print("post-update installed at .git\hooks\post-update")


def post_update_hook():
    build_docs()
