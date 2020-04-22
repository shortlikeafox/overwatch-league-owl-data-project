from os import listdir
from os.path import isfile, join
from bs4 import BeautifulSoup
import pandas as pd
from urllib.request import urlopen
from dateutil.parser import parse


def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try: 
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False

my_path = 'data/odds-scrape/'

files = [f for f in listdir(my_path) if isfile(join(my_path, f))]

for ff in files:
    html=open(my_path + ff, "r")
    bs=BeautifulSoup(html, 'html.parser')
    
    all_elements = bs.find_all()
    
    column_names = []

    df = pd.DataFrame(columns=column_names)
    
    d = ''
    t1 = ''
    t2 = ''
    t1_odds = ''
    t2_odds = ''
    
    for e in all_elements:
        if e.name == 'th':
            #print(e.get('class'))
            if e.get('class') == ['first2', 'tl']:
                #print(e.get_text())
                possible_date = e.get_text()[0:11]
                #print(f"{possible_date}: {is_date(possible_date)}")
                if (is_date(possible_date)):
                    d=possible_date
        elif e.name == 'td':
            if e.get('class') == ['name', 'table-participant']:
                #print(e.get_text())
                teams = [x.strip() for x in e.get_text().split('-')]
                #print(f"team one: {teams[0]} team two: {teams[1]}")
                t1 = teams[0]
                t2 = teams[1]
        elif e.name == 'a':
            #print(e.get('xparam'))
            if e.get('xparam') == 'odds_text':
                if t1_odds == '':
                    
                    t1_odds = e.get_text()
                elif t2_odds == "":
                    t2_odds = e.get_text()
                    a_row = pd.Series([d, t1, t2, t1_odds, t2_odds])
                    row_df = pd.DataFrame([a_row])
                    df = pd.concat([df, row_df])
                    t1 = ''
                    t2 = ''
                    t1_odds = ''
                    t2_odds = ''    
                    
                    
                    
    with open('fight-odds.csv', 'a', newline='') as f:
        df.to_csv(f, header=False)