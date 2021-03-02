from selenium import webdriver
from bs4 import BeautifulSoup

from libs.settings import asx_instruments_url

options = webdriver.ChromeOptions()

options.add_experimental_option('excludeSwitches', ['enable-logging'])  # removes the USB warning on Windows
options.add_argument(
    "--headless"
)  # headless means that no browser window is opened


def get_asx_symbols():
    driver = webdriver.Chrome(options=options)
    driver.get(asx_instruments_url)
    content = driver.page_source
    driver.close()

    soup = BeautifulSoup(content, 'html.parser')
    data = []
    table = soup.find('table', attrs={'class': 'mi-table mt-6'})
    table_body = table.find('tbody')

    rows = table_body.find_all('tr')
    for row in rows:
        cols = row.find_all('td')
        cols = [elem.text.strip() for elem in cols]
        data.append(cols)

    print(f"{len(data)} stocks retreived")

    stocks = [dict(code=elem[2], name=elem[3], price=float(elem[4].replace('$', ''))) for elem in data]
    return stocks