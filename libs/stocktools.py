import requests
import pandas as pd
import io
from libs.settings import asx_instruments_url


def get_asx_symbols():
    source = requests.get(asx_instruments_url).content
    content = pd.read_csv(io.StringIO(source.decode('utf-8')), skiprows=1)
    return content