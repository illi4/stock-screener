import argparse
import arrow

parser = argparse.ArgumentParser()


def define_args():
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update the assets list. Do this before scanning.",
    )
    parser.add_argument(
        "--scan", action="store_true", help="Scan for potential signals"
    )
    parser.add_argument(
        "-exchange",
        type=str,
        required=True,
        help="Exchange (asx|nasdaq|all)",
        choices=["asx", "nasdaq", "all"],
    )
    parser.add_argument(
        "-num", type=int, required=False, help="Limit the number of scanned stocks"
    )
    parser.add_argument(
        "-system",
        type=str,
        required=True,
        help="System (2ma|3ma)",
        choices=["2ma", "3ma"],
    )

    args = parser.parse_args()
    arguments = vars(args)

    if not arguments["update"]:
        arguments["update"] = False
    if not arguments["scan"]:
        arguments["scan"] = False
    arguments["exchange"] = arguments["exchange"].upper()

    if True not in arguments.values():
        print("No arguments specified. Run main.py --h to show help.")
        exit(0)

    return arguments


def dates_diff(date_from, date_to=None):
    date_to = arrow.now() if date_to is None else date_to
    return (date_to - date_from).days


def format_number(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return "%.2f%s" % (num, ["", "K", "M", "G", "T", "P"][magnitude])


def format_bool(value):
    # Format boolean as tick or fail
    formatted_value = "v" if value else "x"
    return formatted_value


def get_test_stocks():
    # Use for testing / bugfixes
    # In main.py, use: stocks = get_test_stocks()

    class Stk:
        code, name = None, None

    test_stock = Stk()
    test_stock.code = "WAM"
    test_stock.name = "WAM"
    return [test_stock]
