import argparse
import arrow

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


def dates_diff(date_from, date_to=None):
    date_to = arrow.now() if date_to is None else date_to
    return (date_to - date_from).days



