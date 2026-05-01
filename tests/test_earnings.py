"""
Test script for Finnhub earnings calendar functions.
Run from project root: FINNHUB_API_KEY=your_key python tests/test_earnings_calendar.py
"""
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from libs.stocktools import get_earnings_calendar, get_upcoming_earnings_symbols


def test_returns_dict():
    date_from = (date.today() - timedelta(days=7)).strftime("%Y-%m-%d")
    date_to = date.today().strftime("%Y-%m-%d")
    result = get_earnings_calendar(date_from, date_to)
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    print(f"[PASS] Returns dict ({len(result)} symbols)")


def test_dict_values_are_date_strings():
    result = get_earnings_calendar("2024-10-01", "2024-10-31")
    assert result, "Expected non-empty result for Oct 2024"
    sample_symbol, sample_date = next(iter(result.items()))
    assert isinstance(sample_date, str) and len(sample_date) == 10, \
        f"Expected YYYY-MM-DD date string, got: {sample_date!r}"
    print(f"[PASS] Dict values are date strings (sample: {sample_symbol} -> {sample_date})")


def test_known_symbols_present():
    result = get_earnings_calendar("2026-05-01", "2026-05-25")
    well_known = {"SNT", "GMHS", "AMZN", "GOOGL", "META"}
    found = well_known & result.keys()
    assert found, f"Expected at least one of {well_known}, got sample: {list(result.keys())[:10]}"
    print(f"[PASS] Known symbols present: {found}")


def test_dates_within_requested_range():
    date_from, date_to = "2026-05-01", "2026-05-25"
    result = get_earnings_calendar(date_from, date_to)
    for symbol, d in result.items():
        assert date_from <= d <= date_to, f"{symbol} has date {d} outside [{date_from}, {date_to}]"
    print(f"[PASS] All returned dates within requested range ({len(result)} entries checked)")


def test_upcoming_returns_set():
    result = get_upcoming_earnings_symbols(days_ahead=7)
    assert isinstance(result, set), f"Expected set, got {type(result)}"
    print(f"[PASS] get_upcoming_earnings_symbols returns set ({len(result)} symbols)")


def test_upcoming_dates_are_in_future():
    """Verify upcoming earnings are indeed from today onwards."""
    today = date.today()
    calendar = get_earnings_calendar(
        today.strftime("%Y-%m-%d"),
        (today + timedelta(days=7)).strftime("%Y-%m-%d")
    )
    for symbol, d in calendar.items():
        assert d >= today.strftime("%Y-%m-%d"), \
            f"{symbol} has past date {d} in upcoming earnings window"
    print(f"[PASS] All upcoming earnings dates are today or future ({len(calendar)} entries)")


if __name__ == "__main__":
    print(f"FINNHUB_API_KEY set: {'Yes' if os.environ.get('FINNHUB_API_KEY') else 'NO — set it first!'}\n")

    tests = [
        test_returns_dict,
        test_dict_values_are_date_strings,
        test_known_symbols_present,
        test_dates_within_requested_range,
        test_upcoming_returns_set,
        test_upcoming_dates_are_in_future,
    ]

    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
        except Exception as e:
            print(f"[ERROR] {t.__name__}: {e}")

    print(f"\n{passed}/{len(tests)} tests passed")