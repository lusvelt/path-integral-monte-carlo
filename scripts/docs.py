import os
import subprocess

root_dir = os.getcwd()
docs_dir = os.path.join(root_dir, "docs")
docs_source_dir = os.path.join(docs_dir, "source")


def generate_docs():
    for filename in os.listdir(docs_source_dir):
        file_path = os.path.join(docs_source_dir, filename)
        if filename.endswith(".rst") and filename != "index.rst":
            os.remove(file_path)
    subprocess.run(["sphinx-apidoc", "-o", "docs/source/", "src"])


def build_docs():
    generate_docs()
    os.chdir(docs_dir)
    if os.name == "nt":
        subprocess.run(["make.bat", "html"])
    elif os.name == "posix":
        subprocess.run(["make", "html"])
    os.chdir(root_dir)
