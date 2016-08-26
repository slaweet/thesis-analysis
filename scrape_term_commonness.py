#!/usr/bin/python
# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import urllib2
import re
import csv
import optparse
import time


def main():
    p = optparse.OptionParser()
    options, arguments = p.parse_args()
    if len(arguments) < 2:
        print 'USAGE: ./scrape_term_commonness.py <input_flashcards.csv> <output_commonness.csv>\n'
    terms_cache = {}
    with open(arguments[1], 'rb') as ofile:
        terms_reader = csv.reader(ofile, delimiter=',')
        for row in terms_reader:
            terms_cache[row[1]] = row[2]
    with open(arguments[0], 'rb') as csvfile:
        with open(arguments[1], 'a') as ofile:
            terms_reader = csv.reader(csvfile, delimiter=',')
            writer = csv.writer(ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
            for row in terms_reader:
                term = row[2]
                if term not in terms_cache:
                    commonness = get_commonness(term.split(';')[0])
                    if commonness is not None:
                        terms_cache[term] = commonness
                        print row[0], ',', term, ',', commonness
                        # terms.append([term, commonness])
                        writer.writerow([row[0], term, commonness])
                        # print term
                    else:
                        print 'commonness failed: ' + term
                # else:
                    # print "Skipped: ", term


def get_commonness(term):
    time.sleep(2)
    page = urllib2.urlopen("https://search.seznam.cz/?q=" + term.replace(' ', '+')).read()
    soup = BeautifulSoup(page)
    div = soup.find("div", {"id": "resultCount"})
    if div is None:
        return None
    count_text = div.findAll("strong")[2].text
    count_text = re.sub(r'\D', '', count_text)
    count = int(count_text)
    return count


if __name__ == '__main__':
    main()
