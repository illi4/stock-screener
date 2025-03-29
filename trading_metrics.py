#!/usr/bin/env python3
"""
Trading Journal Metrics Calculator

This script calculates various trading performance metrics from a Google Sheets trading journal.
It reads data from multiple tabs, filters by date range, and produces detailed performance analysis.
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import yaml
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Default tabs to process
DEFAULT_TABS = [
    "Stocks_ANX_Long",
    "Stocks_E_reversal",
    "Stocks_ANX_Short",
    "Stocks_SAR_MA"
]


def read_config():
    """Read configuration from config.yaml file."""
    try:
        with open("config.yaml", "r") as config_file:
            return yaml.safe_load(config_file)
    except Exception as e:
        print(f"Error reading config file: {e}")
        exit(1)


def connect_to_google_sheets(config):
    """Connect to Google Sheets using credentials."""
    try:
        # Use the credentials file path from config or default to this location
        credentials_path = config.get("logging", {}).get("gsheet_credentials_path",
                                                         ".config/gspread/service_account.json")

        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
        client = gspread.authorize(creds)

        sheet_name = config.get("logging", {}).get("gsheet_name", "Trading journal FY24/25")
        sheet = client.open(sheet_name)

        return sheet
    except Exception as e:
        print(f"Error connecting to Google Sheets: {e}")
        print("Please ensure your credentials file is in the correct location and has the necessary permissions.")
        exit(1)


def get_sheet_tabs(sheet, tab_names=None):
    """Get a list of worksheets from the Google Sheet."""
    all_tab_titles = [ws.title for ws in sheet.worksheets()]

    if tab_names:
        # Return only the specified tabs if they exist
        return [sheet.worksheet(tab) for tab in tab_names if tab in all_tab_titles]
    else:
        # Return default tabs if they exist
        return [sheet.worksheet(tab) for tab in DEFAULT_TABS if tab in all_tab_titles]


def sheet_to_dataframe(worksheet):
    """Convert a Google Sheet worksheet to a pandas DataFrame."""
    try:
        # Get all values including header row
        data = worksheet.get_all_values()

        if not data:
            print(f"No data found in worksheet {worksheet.title}")
            return pd.DataFrame()

        # First row is header
        headers = data[0]

        # Rest of the rows are data
        rows = data[1:]

        # Create DataFrame
        df = pd.DataFrame(rows, columns=headers)

        # Convert date columns to datetime
        for date_col in ['Entry date', 'Exit date']:
            if date_col in df.columns:
                # Try different date formats
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                # For dates in DD/MM/YYYY format
                mask = df[date_col].isna()
                if mask.any():
                    df.loc[mask, date_col] = pd.to_datetime(df.loc[mask, date_col].astype(str), format="%d/%m/%Y",
                                                            errors='coerce')

        # Handle percentage values in Result % column
        if 'Result %' in df.columns:
            df['Result %'] = df['Result %'].astype(str).apply(lambda x: x.replace('%', '').strip())
            df['Result %'] = pd.to_numeric(df['Result %'], errors='coerce') / 100

        # Process numeric columns
        numeric_cols = [
            'Entry price 1 (USD)', 'Position size 1 (stocks)',
            'Entry price 2 (USD)', 'Position size 2 (stocks)',
            'Exit price (USD)', 'Fees (USD)',
            'PNL entry 1 (USD)', 'PNL entry 2 (USD)',
            'Result PNL (USD)',
            'Cumulative PNL (USD)', 'USD to AUD',
            'Result PNL (AUD)'
        ]

        for col in numeric_cols:
            if col in df.columns:
                # Process each value and clean it
                df[col] = df[col].astype(str).apply(lambda x: x.replace('$', '').replace(',', '').strip())
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Special handling for Stop column
        if 'Stop' in df.columns:
            # Try to convert to numeric, setting non-numeric values to NaN
            df['Stop'] = df['Stop'].astype(str).apply(lambda x: x.replace('$', '').replace(',', '').strip())
            df['Stop'] = pd.to_numeric(df['Stop'], errors='coerce')

        return df

    except Exception as e:
        print(f"Error converting worksheet {worksheet.title} to DataFrame: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def filter_trades_by_date(df, start_date, end_date):
    """
    Filter trades by entry date within the specified range and exclude open positions.
    Open positions are identified by having no Exit date.
    """
    if 'Entry date' not in df.columns:
        print("Warning: 'Entry date' column not found in DataFrame")
        return df

    # Convert string dates to datetime if they aren't already
    if not isinstance(start_date, datetime):
        start_date = pd.to_datetime(start_date)
    if not isinstance(end_date, datetime):
        end_date = pd.to_datetime(end_date)

    # Drop rows with invalid entry dates
    df = df.dropna(subset=['Entry date'])

    # Filter by date range
    date_filtered = df[(df['Entry date'] >= start_date) & (df['Entry date'] <= end_date)]

    # Exclude open positions (those without an exit date)
    if 'Exit date' in df.columns:
        closed_positions = date_filtered.dropna(subset=['Exit date'])
        open_positions_count = len(date_filtered) - len(closed_positions)
        if open_positions_count > 0:
            print(f"Excluded {open_positions_count} open positions from analysis")
        return closed_positions
    else:
        print("Warning: 'Exit date' column not found in DataFrame, can't filter open positions")
        return date_filtered


def calculate_drawdown(equity_curve):
    """Calculate the maximum drawdown from an equity curve."""
    # Make a copy of the equity curve and ensure it doesn't have NaN values
    equity = equity_curve.dropna().copy()

    if len(equity) <= 1:
        return 0

    # Calculate running maximum
    running_max = np.maximum.accumulate(equity)

    # Calculate drawdown
    drawdown = (equity - running_max) / running_max

    # Return the maximum drawdown as a percentage (negative value)
    max_dd = drawdown.min()
    return max_dd * 100 if not pd.isna(max_dd) else 0


def calculate_metrics(df):
    """Calculate trading metrics from the filtered DataFrame."""
    if df.empty:
        return {
            'total_trades': 0,
            'win_count': 0,
            'loss_count': 0,
            'win_rate': 0,
            'avg_win_pct': 0,
            'avg_loss_pct': 0,
            'cum_pnl_aud': 0,
            'cum_pnl_pct': 0,
            'profit_factor': 0,
            'expectancy': 0,
            'max_drawdown': 0,
            'avg_r': 0
        }

    # Calculate basic counts
    total_trades = len(df)

    # Check if 'Win/loss' column exists, otherwise use Result % to determine
    if 'Win/loss' in df.columns and not df['Win/loss'].isna().all():
        # Handle string or numeric Win/loss values
        if df['Win/loss'].dtype == 'object':  # String type
            wins = df[df['Win/loss'].str.lower().str.strip() == 'win']
            losses = df[df['Win/loss'].str.lower().str.strip() == 'loss']
        else:  # Numeric type
            wins = df[df['Win/loss'] > 0]
            losses = df[df['Win/loss'] < 0]
    else:
        # Fallback to using Result % if Win/loss column is not available
        wins = df[df['Result %'] > 0]
        losses = df[df['Result %'] < 0]

    win_count = len(wins)
    loss_count = len(losses)

    # Calculate percentages
    win_rate = win_count / total_trades if total_trades > 0 else 0

    # Average win/loss percentages - handle NaN and empty dataframes
    try:
        avg_win_pct = wins['Result %'].mean() if not wins.empty else 0
        if pd.isna(avg_win_pct):  # If still NaN, set to 0
            avg_win_pct = 0
    except:
        avg_win_pct = 0

    try:
        avg_loss_pct = losses['Result %'].mean() if not losses.empty else 0
        if pd.isna(avg_loss_pct):  # If still NaN, set to 0
            avg_loss_pct = 0
    except:
        avg_loss_pct = 0

    # Cumulative P&L - handle NaN
    try:
        cum_pnl_aud = df['Result PNL (AUD)'].sum()
        if pd.isna(cum_pnl_aud):
            cum_pnl_aud = 0
    except:
        cum_pnl_aud = 0

    # Calculate cumulative P&L percentage
    try:
        cum_pnl_pct = df['Result %'].sum()
        if pd.isna(cum_pnl_pct):
            cum_pnl_pct = 0
    except:
        cum_pnl_pct = 0

    # Profit factor - handle edge cases
    try:
        gross_profit = wins['Result PNL (AUD)'].sum() if not wins.empty else 0
        if pd.isna(gross_profit):
            gross_profit = 0

        gross_loss = abs(losses['Result PNL (AUD)'].sum()) if not losses.empty else 0
        if pd.isna(gross_loss):
            gross_loss = 0

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    except:
        profit_factor = 0

    # Expectancy - handle NaN
    try:
        expectancy = (win_rate * avg_win_pct) - ((1 - win_rate) * abs(avg_loss_pct))
        if pd.isna(expectancy):
            expectancy = 0
    except:
        expectancy = 0

    # Calculate drawdown - make more robust
    try:
        # For this, we need to reconstruct the equity curve
        if 'Cumulative PNL (USD)' in df.columns and df['Cumulative PNL (USD)'].notna().any():
            equity_curve = df.sort_values('Entry date')['Cumulative PNL (USD)'].dropna()
        else:
            # Recreate cumulative PNL from individual trade results
            df = df.sort_values('Entry date')
            equity_curve = df['Result PNL (AUD)'].cumsum().dropna()

        # Only calculate drawdown if we have equity data
        if not equity_curve.empty:
            max_drawdown = calculate_drawdown(equity_curve)
        else:
            max_drawdown = 0
    except Exception as e:
        print(f"Error calculating drawdown: {e}")
        max_drawdown = 0

    # Calculate average R (assuming 1R is the distance from entry to initial stop loss)
    avg_r = None
    try:
        if ('Stop' in df.columns and 'Entry price 1 (USD)' in df.columns and
                df['Stop'].notna().any() and df['Position size 1 (stocks)'].notna().any()):

            # Only use rows with valid numeric values for all required fields
            valid_rows = df.dropna(
                subset=['Stop', 'Entry price 1 (USD)', 'Position size 1 (stocks)', 'Result PNL (USD)'])

            if not valid_rows.empty:
                # Calculate initial risk
                valid_rows['initial_risk'] = abs(valid_rows['Entry price 1 (USD)'] - valid_rows['Stop'])

                # Filter out rows with zero or extremely small risk
                valid_rows = valid_rows[valid_rows['initial_risk'] > 0.0001]

                if not valid_rows.empty:
                    # Calculate R multiple
                    valid_rows['r_multiple'] = valid_rows['Result PNL (USD)'] / (
                                valid_rows['initial_risk'] * valid_rows['Position size 1 (stocks)'])

                    # Calculate average R, handling outliers
                    # Trim extreme values (more than 10 standard deviations from mean)
                    r_values = valid_rows['r_multiple']
                    mean_r = r_values.mean()
                    std_r = r_values.std()

                    if not pd.isna(std_r) and std_r > 0:
                        trimmed_r_values = r_values[(r_values > mean_r - 10 * std_r) & (r_values < mean_r + 10 * std_r)]
                        avg_r = trimmed_r_values.mean() if not trimmed_r_values.empty else r_values.mean()
                    else:
                        avg_r = r_values.mean()
    except Exception as e:
        print(f"Error calculating R multiple: {e}")

    return {
        'total_trades': total_trades,
        'win_count': win_count,
        'loss_count': loss_count,
        'win_rate': win_rate,
        'avg_win_pct': avg_win_pct,
        'avg_loss_pct': avg_loss_pct,
        'cum_pnl_aud': cum_pnl_aud,
        'cum_pnl_pct': cum_pnl_pct,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        'max_drawdown': max_drawdown,
        'avg_r': avg_r
    }


def print_metrics(tab_name, metrics):
    """Print trading metrics in a formatted manner."""
    print(f"\n{'=' * 50}")
    print(f"Trading Metrics for Tab: {tab_name}")
    print(f"{'=' * 50}")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Count: {metrics['win_count']}")
    print(f"Loss Count: {metrics['loss_count']}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Average Win: {metrics['avg_win_pct']:.2%}")
    print(f"Average Loss: {metrics['avg_loss_pct']:.2%}")
    print(f"Cumulative P&L (AUD): ${metrics['cum_pnl_aud']:.2f}")
    print(f"Cumulative P&L (%): {metrics['cum_pnl_pct']:.2%}")

    if metrics['avg_r'] is not None:
        print(f"Average R: {metrics['avg_r']:.2f}")
    else:
        print("Average R: Not enough data to calculate")

    print(f"Profit Factor: {metrics['profit_factor']:.2f}")

    # Calculate expectancy as a dollar amount using values from the metrics dictionary
    expectancy_dollar = metrics['win_rate'] * metrics['avg_win_pct'] * 100 - (1 - metrics['win_rate']) * abs(
        metrics['avg_loss_pct']) * 100
    print(f"Expectancy: ${expectancy_dollar:.2f}")

    print(f"Maximum Drawdown: {abs(metrics['max_drawdown']):.2%}")
    print(f"{'=' * 50}\n")


def export_metrics_to_csv(all_metrics, filename="trading_metrics_summary.csv"):
    """Export all metrics to a CSV file."""
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df = metrics_df.reset_index().rename(columns={'index': 'Tab'})

    # Format percentage columns
    percent_cols = ['win_rate', 'avg_win_pct', 'avg_loss_pct', 'cum_pnl_pct', 'expectancy', 'max_drawdown']
    for col in percent_cols:
        metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A")

    # Format currency columns
    currency_cols = ['cum_pnl_aud', 'profit_factor']
    for col in currency_cols:
        metrics_df[col] = metrics_df[col].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A")

    # Format other numeric columns
    numeric_cols = ['total_trades', 'win_count', 'loss_count', 'avg_r']
    for col in numeric_cols:
        metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")

    metrics_df.to_csv(filename, index=False)
    print(f"Metrics exported to {filename}")


def main():
    """Main function to process trading journal and calculate metrics."""
    parser = argparse.ArgumentParser(description='Calculate trading metrics from a Google Sheets trading journal.')
    parser.add_argument('--start_date', required=True, help='Start date for filtering trades (YYYY-MM-DD)')
    parser.add_argument('--end_date', required=True, help='End date for filtering trades (YYYY-MM-DD)')
    parser.add_argument('--tabs', nargs='+', help='Specific tabs to process (defaults to predefined trading tabs)')
    parser.add_argument('--export', action='store_true', help='Export metrics to CSV')

    args = parser.parse_args()

    # Validate dates
    try:
        start_date = pd.to_datetime(args.start_date)
        end_date = pd.to_datetime(args.end_date)
    except ValueError:
        print("Error: Invalid date format. Please use YYYY-MM-DD.")
        return

    # Read configuration
    config = read_config()

    # Connect to Google Sheets
    sheet = connect_to_google_sheets(config)

    # Get tabs to process (using default tabs if none specified)
    tabs = get_sheet_tabs(sheet, args.tabs)

    print(f"Processing trades from {args.start_date} to {args.end_date}")
    if args.tabs:
        print(f"Processing user-specified tabs: {', '.join(args.tabs)}")
    else:
        print(f"Processing default tabs: {', '.join(DEFAULT_TABS)}")
    print(f"Found {len(tabs)} valid tabs to process")

    all_metrics = {}

    # Process each tab
    for tab in tabs:
        print(f"\nProcessing tab: {tab.title}")

        # Read data from the tab
        df = sheet_to_dataframe(tab)

        if df.empty:
            print(f"No data found in tab: {tab.title}")
            continue

        # Filter by date range
        filtered_df = filter_trades_by_date(df, start_date, end_date)

        print(f"Found {len(filtered_df)} trades in date range")

        # Calculate metrics
        metrics = calculate_metrics(filtered_df)
        all_metrics[tab.title] = metrics

        # Print metrics
        print_metrics(tab.title, metrics)

    # Export metrics if requested
    if args.export and all_metrics:
        export_metrics_to_csv(all_metrics)

    # Print overall summary
    if all_metrics:
        # Calculate aggregate metrics across all tabs
        total_trades = sum(m['total_trades'] for m in all_metrics.values())
        total_wins = sum(m['win_count'] for m in all_metrics.values())
        total_losses = sum(m['loss_count'] for m in all_metrics.values())
        overall_win_rate = total_wins / total_trades if total_trades > 0 else 0

        print("\n" + "=" * 50)
        print("OVERALL SUMMARY")
        print("=" * 50)
        print(f"Total Tabs Processed: {len(all_metrics)}")
        print(f"Total Trades: {total_trades}")
        print(f"Total Wins: {total_wins}")
        print(f"Total Losses: {total_losses}")
        print(f"Overall Win Rate: {overall_win_rate:.2%}")
        print("=" * 50)


if __name__ == "__main__":
    main()