# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import load_data

DATA_DIR = 'data/'
PLOT_DIR = 'plot/'


class Command(object):
    name = 'command'

    def __init__(self, options):
        self.options = options

    def generate_graph(self, data):
        print len(data)
        print data.head()
        data.plot(kind='bar')
        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.3)
        plt.savefig(PLOT_DIR + self.name + '.png')
        plt.show()

    def execute(self):
        data = self.get_data()
        self.generate_graph(data)

    def get_data(self):
        pass


class MnemonicsEffect(Command):
    name = 'mnemonics_effect'

    def get_data(self):
        mnemonics = load_data.get_mnemonics(self.options)
        answers = load_data.get_answers_with_map(self.options)
        answers = answers.sort(['place_asked', 'user'], ascending=True)
        answers = answers.reset_index()
        answers['seen_times'] = 0
        for index, row in answers.iterrows():
            if index != 0:
                prew = answers.loc[[index - 1]]
                # print row['user'], prew['user'].values[0], prew['seen_times'].values[0]
                # print row['user'] == prew['user'].values[0]
                # print row['place_asked'] == prew['place_asked'].values[0]
                if row['place_asked'] == prew['place_asked'].values[0] and row['user'] == prew['user'].values[0]:
                    answers['seen_times'][index] = prew['seen_times']
                    answers['seen_times'][index] += 1
        # answers['seen_times'] = answers['seen_times'].shift(1) + 1 if (
            # answers['user'].shift(1) == answers['user']) else 0
        print answers[['user', 'seen_times']]
        mnemonics_added_on = '2014-12-20 06:23:00'
        answers['after_mnemonics'] = answers['inserted'] > mnemonics_added_on
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
    name = 'filter_lithuania'

    def get_data(self):
        answers = load_data.get_answers(self.options)
        answers = answers[answers.place_asked == '142']
        print answers
        answers.to_csv('data/answers-lithuania.csv', sep=',', encoding='utf-8')


class FilterEuropeStates(Command):
    name = 'filter_europe_states'

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
    name = 'rating_by_map'

    def get_data(self):
        ratings = load_data.get_rating(self.options)
        return ratings


class PlacesByMap(Command):
    name = 'places_by_map'

    def get_data(self):
        maps = load_data.get_maps(self.options)
        maps = maps.groupby(['map_name']).count()
        maps = maps[['place']]
        maps = maps.sort('place', ascending=False)
        maps.columns = ['Number Of Places']
        return maps


class PlacesByMapAndType(Command):
    name = 'places_by_map_and_type'

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
    name = 'success_by_map'

    def get_data(self):
        answers_with_maps = load_data.get_answers_with_map_grouped(self.options)
        success_rate_by_map = answers_with_maps.mean().sort('correct')
        return success_rate_by_map
