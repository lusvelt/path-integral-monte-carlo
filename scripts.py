import os
import argparse
import subprocess

root_dir = os.getcwd()
docs_dir = os.path.join(root_dir, 'docs')
docs_source_dir = os.path.join(docs_dir, 'source')
environment_yml_file = os.path.join(root_dir, 'environment.yml')
hooks_dir = os.path.join(root_dir, '.git', 'hooks')


def generate_docs():
    for filename in os.listdir(docs_source_dir):
        file_path = os.path.join(docs_source_dir, filename)
        if filename.endswith('.rst') and filename != 'index.rst':
            os.remove(file_path)
    subprocess.run(['sphinx-apidoc', '-o', 'docs/source/', 'src'])


def build_docs():
    generate_docs()
    os.chdir(docs_dir)
    if os.name == 'nt':
        subprocess.run(['make.bat', 'html'])
    elif os.name == 'posix':
        subprocess.run(['make', 'html'])
    os.chdir(root_dir)


def export_environment():
    result = subprocess.run(['conda', 'env', 'export'], capture_output=True, text=True)
    env_str = result.stdout
    lines = env_str.splitlines(keepends=True)
    lines[0] = 'name: lqfn\n'
    lines.pop()
    with open('environment.yml', 'w') as file:
        file.writelines(lines)
        file.close()


def import_environment():
    subprocess.run(['conda', 'env', 'update', '--file', 'environment.yml'])


# Git hooks
def install_hooks():
    subprocess.run(['pre-commit', 'install'])
    post_update_path = os.path.join(hooks_dir, 'post-update')
    with open(post_update_path, 'w') as file:
        file.write("#!/bin/bash\npython scripts.py post-update-hook\n")
        file.close()
    file.close()
    os.chmod(post_update_path, 0o755)
    print('post-update installed at .git\hooks\post-update')


def pre_commit_custom_hook():
    export_environment()
    os.system('git add .')


def post_update_hook():
    import_environment()


# General install
def install():
    install_hooks()
    build_docs()
    subprocess.run(['pip', 'install', '-e', '.'])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Useful scripts for the project')

    subparsers = parser.add_subparsers(dest='command')

    build_docs_parser = subparsers.add_parser('build-docs', help='Generates sphinx documentation in docs/source folder and builds html docs in docs/build folder. Open docs/build/index.html in browser to see the documentation.')
    export_environment_parser = subparsers.add_parser('export-environment', help='Exports current anaconda environment to environment.yml file')
    import_environment_parser = subparsers.add_parser('import-environment', help='Imports anaconda environment from environment.yml file into the current anaconda environment')

    install_hooks_parser = subparsers.add_parser('install-hooks', help='Installs hooks in .git/hooks folder')
    pre_commit_custom_hook_parser = subparsers.add_parser('pre-commit-custom-hook', help='Executes the custom pre-commit hook (the standard tasks of pre-commit package are not executed)')
    post_update_hook_parser = subparsers.add_parser('post-update-hook', help='Executes post-update hook')

    install_parser = subparsers.add_parser('install', help='Performs first installation tasks')

    args, _ = parser.parse_known_args()

    if args.command == 'build-docs':
        generate_docs()
    elif args.command == 'export-environment':
        export_environment()
    elif args.command == 'import-environment':
        import_environment()
    elif args.command == 'install-hooks':
        install_hooks()
    elif args.command == 'pre-commit-custom-hook':
        pre_commit_custom_hook()
    elif args.command == 'post-update-hook':
        post_update_hook()
    elif args.command == 'install':
        install()
    else:
        parser.print_help()
