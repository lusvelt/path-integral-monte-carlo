import os
import subprocess
import argparse

from scripts.conda_tools import conda_export


def export_environment():
    os.system('conda activate lqfn')
    args = argparse.Namespace(from_history=True, env_name='lqfn', output='environment.yml', no_builds=True, use_versions=False, verbose=False, include_prefix=False)
    conda_export.main(args)


def import_environment():
    subprocess.run(['conda', 'env', 'update', '--file', 'environment.yml'])
