from libs.stocktools import get_asx_symbols
from pprint import pprint

from bs4 import BeautifulSoup

# Have to use selenium hey


#import yfinance as yf
#asset = yf.Ticker("BPP.AX")  # add .AX for aussie stocks
# 2k limit for the asx
# get stock info
#pprint(msft.info)

#hist = asset.history(period='90d', interval='1d').reset_index(drop=False)
#print(hist)

#a = get_asx_symbols()
#print(a)

# Test scraping
'''
import requests
page = requests.get("https://www.marketindex.com.au/asx-listed-companies")
soup = BeautifulSoup(page.content, 'html.parser')
#print(soup.prettify())

tables = soup.findChildren('table')
print(tables)

#table = soup.find_all('table') # , attrs={'class':'mi-table mt-6'})

#print(list(soup.children))

exit(0)

table_body = table.find('tbody')

data = []
rows = table_body.find_all('tr')
for row in rows:
    cols = row.find_all('td')
    cols = [ele.text.strip() for ele in cols]
    data.append([ele for ele in cols if ele]) # Get rid of empty values

print(data)
'''

# Selenium plus soup
from selenium import webdriver
options = webdriver.ChromeOptions()

options.add_experimental_option('excludeSwitches', ['enable-logging'])
options.add_argument(
    "--headless"
)  # headless means that no browser window is opened

driver = webdriver.Chrome(options=options)


driver.get("https://www.marketindex.com.au/asx-listed-companies")
content = driver.page_source
driver.close()

soup = BeautifulSoup(content, 'html.parser')
data = []
table = soup.find('table', attrs={'class':'mi-table mt-6'})
table_body = table.find('tbody')

rows = table_body.find_all('tr')
for row in rows:
    cols = row.find_all('td')
    cols = [ele.text.strip() for ele in cols]
    data.append([ele for ele in cols if ele]) # Get rid of empty values

print(data) # cool