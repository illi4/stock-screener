import argparse
import arrow

parser = argparse.ArgumentParser()


def get_data_start_date(input_date=None):
    if input_date is None:
        current_date = arrow.now()
    else:
        current_date = arrow.get(input_date.strftime("%Y-%m-%d"), "YYYY-MM-DD")

    shifted_date = current_date.shift(months=-12)
    data_start_date = shifted_date.format("YYYY-MM-DD")

    return data_start_date


def get_previous_workday():
    current_datetime = arrow.now()
    current_dow = current_datetime.isoweekday()
    if current_dow == 1:  # only subtract if today is Monday
        current_datetime = current_datetime.shift(days=-3)
    else:
        current_datetime = current_datetime.shift(days=-1)
    current_datetime = current_datetime.format("YYYY-MM-DD")
    return current_datetime


def get_current_workday():
    current_datetime = arrow.now()
    current_datetime = current_datetime.format("YYYY-MM-DD")
    return current_datetime


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
        "-date",
        type=str,
        required=False,
        help="Date to run as of (YYYY-MM-DD format) for update or scan",
    )
    parser.add_argument(
        "-num", type=int, required=False, help="Limit the number of scanned stocks"
    )
    parser.add_argument(
        "-method",
        type=str,
        required=True,
        choices=["mri", "anx"],
        help="Method of shortlisting (mri or anx)"
    )

    args = parser.parse_args()
    arguments = vars(args)

    if not arguments["update"]:
        arguments["update"] = False
    if not arguments["scan"]:
        arguments["scan"] = False

    # Process the date
    if arguments["date"] is not None:
        try:
            arguments["date"] = arrow.get(arguments["date"], "YYYY-MM-DD").naive
        except arrow.parser.ParserMatchError:
            print("The date must be in the format YYYY-MM-DD")
            exit(0)

    if True not in arguments.values():
        print("No arguments specified. Run scanner.py --h to show help.")
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
    # In scanner.py, use: stocks = get_test_stocks()

    class Stk:
        code, name = None, None

    test_stock = Stk()
    test_stock.code = "ARU"
    test_stock.name = "ARU"
    return [test_stock]
