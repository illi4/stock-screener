import gspread
import pandas as pd

# Init objects to work with the sheets
try:
    gc = gspread.service_account(filename=".config/gspread/service_account.json")
except FileNotFoundError:
    print(
        "Please save your Google service account credentials under .config/gspread/service_account.json"
    )
    exit(0)


def sheet_to_df(book_name, sheet_name):
    """
    Reads data from the named gsheet to df
    :return: pandas dataframe
    """
    df = None
    sh = gc.open(book_name)
    worksheet = sh.worksheet(sheet_name)
    header_row = worksheet.row_values(1)
    values = worksheet.get_all_values()
    values.pop(0)  # get all values and remove the first (header)
    df = pd.DataFrame(values)
    df.columns = header_row

    return df


def sheet_update(book_name, sheet_name, row_idx, column_idx, value):
    sh = gc.open(book_name)
    worksheet = sh.worksheet(sheet_name)
    worksheet.update(f"{column_idx}{row_idx}", value)
