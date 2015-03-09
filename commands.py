# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import load_data
import divider
import pandas as pd
import re
from main import all_subclasses

DATA_DIR = 'data/'
PLOT_DIR = 'plot/'


class Command(object):
    kind = 'bar'

    @staticmethod
    def name(self):
        return convert_from_cammel_case(self.__name__)

    def __init__(self, options):
        self.options = options

    def generate_graph(self, data):
        print len(data)
        print data.head()
        data.plot(kind=self.kind)
        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.3)
        plt.savefig(self.file_name())
        plt.show()

    def file_name(self):
        return (PLOT_DIR + self.name(self.__class__) + '_' +
                self.options.answers.replace('data/', '').replace('.csv', '') +
                '.png')

    def execute(self):
        data = self.get_data()
        self.generate_graph(data)

    def get_data(self):
        pass


class MnemonicsEffect(Command):
    seen_times = {}

    def increment_seen(self, row):
        user = row['user']
        place = row['place_asked']
        if user not in self.seen_times:
            self.seen_times[user] = {}
        if place not in self.seen_times[user]:
            self.seen_times[user][place] = 0
        self.seen_times[user][place] += 1
        return self.seen_times[user][place] - 1

    def add_see_times(self, answers):
        answers['seen_times'] = 0
        for index, row in answers.iterrows():
            answers.loc[index, "seen_times"] = self.increment_seen(row)
        # print answers[['user', 'seen_times']]
        return answers

    def get_data(self):
        answers = load_data.get_answers_with_map(self.options)
        answers = answers.sort(['place_asked', 'user'], ascending=True)
        answers = answers.reset_index()
        answers = self.add_see_times(answers)
        answers = answers[answers['seen_times'] == 1]

        MNEMONICS_ADDED_ON = '2014-12-20 06:23:00'
        mnemonics = load_data.get_mnemonics(self.options)
        answers['after_mnemonics'] = answers['inserted'] > MNEMONICS_ADDED_ON
        answers['has_mnemonics'] = answers['place_asked'].isin(mnemonics)
        answers = answers.groupby(['place_name', 'after_mnemonics', 'has_mnemonics']).mean()
        answers = answers.reset_index()
        answers = answers.pivot(
            index='place_name',
            columns='after_mnemonics',
            values='correct')

        answers['diff'] = answers[False] - answers[True]
        answers = answers.sort('diff', ascending=False)
        answers = answers.drop('diff', 1)
        answers.rename(
            columns={
                True: 'After adding mnemonics',
                False: 'Before adding mnemonics',
            }, inplace=True)
        return answers


class FilterLithuania(Command):
    def get_data(self):
        answers = load_data.get_answers(self.options)
        answers = answers[answers.place_asked == '142']
        print answers
        answers.to_csv('data/answers-lithuania.csv', sep=',', encoding='utf-8')


class FilterEuropeStates(Command):
    def get_data(self):
        maps = load_data.get_maps(self.options)
        maps = maps[maps.place_type == '1']
        maps = maps[maps.map_name == 'Europe']
        answers = load_data.get_answers(self.options)
        print len(answers)
        answers = answers[answers.place_asked.isin(maps.place)]
        print len(answers)
        answers.to_csv('data/answers-europe.csv', sep=',', encoding='utf-8')


class RatingByMap(Command):
    def get_data(self):
        ratings = load_data.get_rating(self.options)
        return ratings


class PlacesByMap(Command):
    def get_data(self):
        maps = load_data.get_maps(self.options)
        maps = maps.groupby(['map_name']).count()
        maps = maps[['place']]
        maps = maps.sort('place', ascending=False)
        maps.columns = ['Number Of Places']
        return maps


class PlacesByMapAndType(Command):
    def get_data(self):
        maps = load_data.get_maps(self.options)
        maps = maps.groupby(['map_name', 'place_type']).count()
        maps = maps.drop(['map_name', 'place_type', 'map', 'place_name'], 1)
        maps = maps.reset_index()
        maps = maps.pivot(
            index='map_name',
            columns='place_type',
            values='place')
        # print maps
        maps = maps[['1', '2', '5', '6', '13', '14']]
        maps.rename(
            columns={
                '1': 'States',
                '2': 'Cities',
                '5': 'Rivers',
                '6': 'Lakes',
                '13': 'Mountains',
                '14': 'Islands',
            }, inplace=True)
        # maps = maps.sort('place', ascending=False)
        return maps


class SuccessByMap(Command):
    def get_data(self):
        answers_with_maps = load_data.get_answers_with_map_grouped(self.options)
        success_rate_by_map = answers_with_maps.mean().sort('correct')
        return success_rate_by_map


class Division(Command):

    def file_name(self):
        return (PLOT_DIR + self.name(self.__class__) +
                '_' + self.options.divider +
                '_' + self.options.answers.replace('data/', '').replace('.csv', '') +
                '.png')

    def get_data(self):
        possible_dividers = dict([
            (Command.name(c), c) for c in all_subclasses(divider.Divider)])
        answers = load_data.get_answers(self.options)
        if not self.options.divider in possible_dividers:
            raise Exception('Invalid divider name: ' + self.options.divider)
        div = possible_dividers[self.options.divider]()
        counts = {}
        for t in range(div.min_treshold, div.max_treshold, 1):
            new_column_name = 'is_school_' + str(t)
            answers_enriched = div.divide(answers, new_column_name, t)
            answers_enriched = answers_enriched.groupby(new_column_name).count()
            answers_enriched = answers_enriched[['id']]
            answers_enriched = answers_enriched.reset_index()
            values = answers_enriched.values
            sum = values[0][1] + (values[1][1] if len(values) > 1 else 0)
            ratio = (values[0][1] * 1.0) / sum
            counts[t] = ratio
        data = pd.Series(counts)
        return data


def convert_from_cammel_case(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
