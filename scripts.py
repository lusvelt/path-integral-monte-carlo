import argparse

from scripts import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Useful scripts for the project")

    subparsers = parser.add_subparsers(dest="command")

    build_docs_parser = subparsers.add_parser(
        "build-docs",
        help="Generates sphinx documentation in docs/source folder and builds html docs in docs/build folder. Open docs/build/index.html in browser to see the documentation.",
    )
    export_environment_parser = subparsers.add_parser(
        "export-environment",
        help="Exports current anaconda environment to environment.yml file",
    )
    import_environment_parser = subparsers.add_parser(
        "import-environment",
        help="Imports anaconda environment from environment.yml file into the current anaconda environment",
    )

    install_hooks_parser = subparsers.add_parser("install-hooks", help="Installs hooks in .git/hooks folder")
    post_update_hook_parser = subparsers.add_parser("post-update-hook", help="Executes post-update hook")

    install_parser = subparsers.add_parser("install", help="Performs first installation tasks")

    args, _ = parser.parse_known_args()

    if args.command == "build-docs":
        build_docs()
    elif args.command == "export-environment":
        export_environment()
    elif args.command == "import-environment":
        import_environment()
    elif args.command == "install-hooks":
        install_hooks()
    elif args.command == "post-update-hook":
        post_update_hook()
    elif args.command == "install":
        install()
    else:
        parser.print_help()
