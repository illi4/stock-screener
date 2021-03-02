import argparse

parser = argparse.ArgumentParser()


def define_args():
    parser.add_argument(
        "--update", action="store_true", help="Update the assets list. Do this before scanning."
    )
    parser.add_argument(
        "--scan", action="store_true", help="Scan for potential signals"
    )

    args = parser.parse_args()
    arguments = vars(args)

    if not arguments["update"]:
        arguments["update"] = False
    if not arguments["scan"]:
        arguments["scan"] = False

    if True not in arguments.values():
        print("No arguments specified. Run main.py --h to show help.")
        exit(0)

    return arguments


