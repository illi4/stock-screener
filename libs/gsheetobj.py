import gspread
import pandas as pd
from string import ascii_uppercase

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


def sheet_update_by_column_name(book_name, sheet_name, row_idx, column_name, value):
    """
    Updates a cell in a Google Sheet using column name instead of column identifier
    """
    sh = gc.open(book_name)
    worksheet = sh.worksheet(sheet_name)

    # Get all column names
    header_row = worksheet.row_values(1)

    # Find the index of the column name
    try:
        column_index = header_row.index(column_name) + 1
    except ValueError:
        raise ValueError(f"Column '{column_name}' not found in the sheet.")

    # Convert column index to letter (A, B, C, ..., AA, AB, etc.)
    column_letter = ''
    while column_index > 0:
        column_index, remainder = divmod(column_index - 1, 26)
        column_letter = ascii_uppercase[remainder] + column_letter

    # Update the cell
    worksheet.update(f"{column_letter}{row_idx}", value)
