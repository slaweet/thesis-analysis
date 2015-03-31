# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import load_data
import divider
import pandas as pd
import utils
import datetime
import math

DATA_DIR = 'data/'
PLOT_DIR = 'plot/'
PARTIAL_DATA_PLOT_DIR = PLOT_DIR + 'partial_data/'

PLACE_TYPES = {
    0: 'unknown',
    1: 'state',
    2: 'city',
    3: 'world',
    4: 'continent',
    5: 'river',
    6: 'lake',
    7: 'region_cz',
    8: 'bundesland',
    9: 'province',
    10: 'region_it',
    11: 'region',
    12: 'autonumous_comunity',
    13: 'mountains',
    14: 'island'
}

LANGUAGES = {
    '0': 'Czech',
    '1': 'English',
    '2': 'Spanish',
}

RATING_VALUES = {
    1: 'Too easy',
    2: 'Appropriate',
    3: 'Too difficult',
}


class Command(object):
    kind = 'bar'
    subplots = False
    adjust_bottom = 0.1
    legend_alpha = False
    ylim = None
    legend_loc = None
    legend_bbox = None
    active = True
    rot = None
    stacked = None
    week_cache = {}

    @staticmethod
    def name(self):
        return utils.convert_from_cammel_case(self.__name__)

    def __init__(self, options, show_plots=True):
        self.options = options
        self.show_plots = show_plots

    def generate_graph(self, data):
        print len(data)
        print data.head()
        data.plot(
            kind=self.kind,
            subplots=self.subplots,
            title=self.plot_name(),
            rot=self.rot,
            stacked=self.stacked,
        )
        fig = plt.gcf()
        if self.ylim is not None:
            axes = plt.gca()
            axes.set_ylim(self.ylim)

        fig.subplots_adjust(bottom=self.adjust_bottom)
        ax = fig.add_subplot(111)
        if self.legend_loc is not None:
            legend = ax.legend(loc=self.legend_loc)
        elif self.legend_bbox is not None:
            legend = ax.legend(bbox_to_anchor=self.legend_bbox)
        else:
            legend = ax.legend()
        if self.legend_alpha:
            legend.get_frame().set_alpha(0.8)

        plt.savefig(self.file_name())
        if self.show_plots:
            plt.show()
        plt.clf()

    def file_name(self):
        if self.options.answers == DATA_DIR + 'geography.answer.csv':
            dest_dir = PLOT_DIR
        else:
            dest_dir = PARTIAL_DATA_PLOT_DIR
        return (dest_dir + utils.convert_from_cammel_case(
            self.plot_name()
        ).replace(' ', '_') + '.png')

    def plot_name(self):
        return (self.__class__.__name__ + ' ' +
                self.options.answers.replace('data/', '').replace('.csv', ''))

    def execute(self):
        if self.options.divider == 'all':
            self.show_plots = False
            divs = divider.Divider.get_divider(self.options)
            for key, div in divs.iteritems():
                print 'processing divider', key
                self.options.divider = key
                self._execute()
        else:
            self._execute()

    def _execute(self):
        data = self.get_data()
        self.generate_graph(data)

    def get_week(self, datetime_string):
        date = datetime_string[:10]
        if date in self.week_cache:
            return self.week_cache[date]
        week = datetime.datetime.strptime(
            date, "%Y-%m-%d"
        ).date().isocalendar()[1]
        if week == 1 and datetime_string[5:7] == '12':
            week = 53
        self.week_cache[date] = week
        return week

    def get_weekday(self, datetime_string):
        date = datetime_string[:10]
        if date in self.week_cache:
            return self.week_cache[date]
        weekday = datetime.datetime.strptime(
            date, "%Y-%m-%d"
        ).date().weekday()
        self.week_cache[date] = weekday
        return weekday

    def add_week(self, answers, field_name):
        answers[field_name] = answers['inserted'].map(
            lambda x:
            x[:5] + 'w' +
            str(self.get_week(x)).zfill(2))
        return answers

    def add_weekday(self, answers, field_name):
        answers[field_name] = answers['inserted'].map(
            lambda x: str(self.get_weekday(x)))
        return answers

    def add_weekday_and_time(self, answers, field_name):
        answers[field_name] = answers['inserted'].map(
            lambda x: str(self.get_weekday(x)) + '-' + x[11:13])
        return answers

    def get_data(self):
        pass


class DivisionCommand(Command):

    def plot_name(self):
        return (self.__class__.__name__ + ' ' +
                self.options.divider + ' ' +
                self.options.answers.replace('data/', '').replace('.csv', ''))


class MnemonicsEffect(Command):
    active = False
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
    active = False

    def get_data(self):
        answers = load_data.get_answers(self.options)
        answers = answers[answers.place_asked == 142]
        print answers
        answers.to_csv('data/answers-lithuania.csv', sep=',', encoding='utf-8')
        return answers


class FilterEuropeStates(Command):
    active = False

    def get_data(self):
        maps = load_data.get_maps(self.options)
        maps = maps[maps.place_type == '1']
        maps = maps[maps.map_name == 'Europe']
        answers = load_data.get_answers(self.options)
        print len(answers)
        answers = answers[answers.place_asked.isin(maps.place)]
        print len(answers)
        answers.to_csv('data/answers-europe.csv', sep=',', encoding='utf-8')
        return answers


class PlacesByMap(Command):
    adjust_bottom = 0.3

    def get_data(self):
        maps = load_data.get_maps(self.options)
        maps = maps.groupby(['map_name']).count()
        maps = maps[['place']]
        maps = maps.sort('place', ascending=False)
        maps.columns = ['Number Of Places']
        return maps


class PlacesByMapAndType(Command):
    adjust_bottom = 0.3

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


class Division(DivisionCommand):
    legend_loc = 'best'
    ylim = [0, 1]

    def get_data(self):
        answers = load_data.get_answers(self.options)
        div = divider.Divider.get_divider(self.options)
        counts = {}
        for t in range(div.min_treshold, div.max_treshold, div.step):
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


class AnswersByMap(DivisionCommand):
    #  kind = 'pie'
    # subplots = True
    adjust_bottom = 0.3

    def get_data(self):
        answers_with_maps = load_data.get_answers_with_map(self.options)
        div = divider.Divider.get_divider(self.options)
        new_column_name = div.column_name
        answers_with_maps = div.divide(answers_with_maps, new_column_name)
        answers_with_maps = answers_with_maps.groupby(['map_name', new_column_name])
        answers_by_map = answers_with_maps.count()
        answers_by_map = answers_by_map[['place']]
        answers_by_map = answers_by_map.reset_index()
        answers_by_map = answers_by_map.pivot(
            index='map_name',
            columns=new_column_name,
            values='place')
        answers_by_map[False] = answers_by_map[False] / answers_by_map[False].sum()
        answers_by_map[True] = answers_by_map[True] / answers_by_map[True].sum()
        answers_by_map['Diff'] = answers_by_map[True] - answers_by_map[False]
        answers_by_map = answers_by_map.sort(False, ascending=False)
        answers_by_map = answers_by_map[:10]
        answers_by_map = answers_by_map.sort('Diff', ascending=False)
        #  answers_by_map.columns = ['Number Of Answers']
        return answers_by_map


class InTimeCommand(Command):
    kind = 'area'
    stacked = True
    legend_alpha = True
    ylim = [0, 1]
    groupby_column = None
    result_columns = None
    drop_duplicate = None
    date_precision = 10
    columns_rename = None
    rot = 90
    adjust_bottom = 0.25
    answers = None
    absolute_values = False
    date_offset = 0

    def setup(self):
        if self.absolute_values:
            self.legend_loc = 'best'
            if self.kind == 'area':
                self.kind = 'line'
            self.stacked = False
            self.ylim = None

    def get_answers(self):
        if self.answers is not None:
            answers = self.answers
        else:
            answers = load_data.get_answers_with_map(self.options)
        if self.date_precision == 'week':
            answers = self.add_week(answers, 'date')
        elif self.date_precision == 'weekday':
            answers = self.add_weekday(answers, 'date')
        elif self.date_precision == 'weekday_and_time':
            answers = self.add_weekday_and_time(answers, 'date')
        else:
            answers['date'] = answers['inserted'].map(
                lambda x: x[self.date_offset:
                            self.date_offset + self.date_precision])
        if self.drop_duplicate is not None:
            answers = answers.drop_duplicates(
                [self.drop_duplicate, 'date', self.groupby_column])
        return answers

    def get_data(self):
        self.setup()
        answers = self.get_answers()
        grouped = answers.groupby(['date', self.groupby_column]).count()
        grouped = grouped[['id']]
        grouped = grouped.reset_index()
        grouped = grouped.pivot(
            index='date',
            columns=self.groupby_column,
            values='id')
        if self.result_columns is None:
            self.result_columns = grouped.columns
        value_columns = grouped.columns
        print grouped
        grouped = grouped.fillna(0)
        if not self.absolute_values:
            grouped['All'] = 0
            for c in value_columns:
                grouped['All'] += grouped[c]
            for c in value_columns:
                grouped[c] = grouped[c] / grouped['All']
            grouped = grouped[grouped['All'] > 10]
        grouped = grouped[self.result_columns]
        if self.columns_rename is not None:
            grouped.rename(columns=self.columns_rename, inplace=True)
        return grouped


class AnswersByMapInTime(InTimeCommand):
    result_columns = [
        'World', 'Europe', 'Africa', 'Asia', 'North America',
        'South America', 'Czech Rep.', 'United States']
    groupby_column = 'map_name'
    date_precision = 'week'


class AnswersByMapInDay(AnswersByMapInTime):
    date_precision = 2
    date_offset = 11


class AnswersByMapInDayAbsolute(AnswersByMapInDay):
    absolute_values = True


class UsersByMapInTime(AnswersByMapInTime):
    drop_duplicate = 'user'


class AnswersByLangInTime(InTimeCommand):
    legend_loc = 'lower left'
    groupby_column = 'language'
    columns_rename = LANGUAGES


class AnswersByLangInDay(AnswersByLangInTime):
    date_precision = 2
    date_offset = 11


class AnswersInDayAbsolute(AnswersByLangInTime):
    result_columns = ['0']
    columns_rename = {'0': 'Number of answers'}
    legend_loc = 'lower right'
    date_precision = 2
    date_offset = 11
    absolute_values = True
    adjust_bottom = 0.1


class AnswersInDayByMinuteAbsolute(AnswersInDayAbsolute):
    date_precision = 5


class AnswersByWeekdayAbsolute(AnswersInDayAbsolute):
    date_precision = 'weekday'
    date_offset = 0


class AnswersByWeekdayAndTimeAbsolute(AnswersInDayAbsolute):
    date_precision = 'weekday_and_time'
    date_offset = 0


class UsersInDayAbsolute(AnswersInDayAbsolute):
    columns_rename = {'0': 'Number of users'}
    drop_duplicate = 'user'


class UsersInDayByMinuteAbsolute(AnswersInDayAbsolute):
    columns_rename = {'0': 'Number of users'}
    drop_duplicate = 'user'
    date_precision = 5


class AnswersInTimeAbsolute(AnswersByLangInTime):
    result_columns = ['0']
    columns_rename = {'0': 'Number of answers'}
    legend_loc = 'lower right'
    date_precision = 7
    absolute_values = True
    adjust_bottom = 0.1


class UsersInTimeAbsolute(AnswersInTimeAbsolute):
    columns_rename = {'0': 'Number of users'}
    drop_duplicate = 'user'


class AnswersByPlaceTypeInTime(InTimeCommand):
    legend_loc = 'lower left'
    groupby_column = 'place_type'
    columns_rename = PLACE_TYPES
    result_columns = [1, 2, 5, 7, 13, 14]
    date_precision = 'week'


class AnswersByNumberOfOptionsInTime(InTimeCommand):
    legend_loc = 'lower right'
    groupby_column = 'number_of_options'


class CorrectRateInTime(InTimeCommand):
    legend_loc = 'upper right'
    groupby_column = 'correct'
    date_precision = 9


class DividerSimilarity(Command):
    adjust_bottom = 0.4

    def get_data(self):
        answers = load_data.get_answers(self.options)
        self.options.divider = 'all'
        divs = divider.Divider.get_divider(self.options)
        for key, div in divs.iteritems():
            answers = div().divide(answers, key)
        grouped = answers.groupby(divs.keys()).count()
        grouped = grouped[['id']]
        # self.options.divider = 'hack'
        return grouped


class DividerCorrelation(Command):
    adjust_bottom = 0.3

    def get_data(self):
        answers = load_data.get_answers(self.options)
        self.options.divider = 'all'
        divs = divider.Divider.get_divider(self.options)
        for key, div in divs.iteritems():
            answers = div().divide(answers, key)
        keys = divs.keys()
        values = answers[keys]
        corr = values.corr()
        return corr


class DivisionInTime(InTimeCommand):

    def plot_name(self):
        return (self.__class__.__name__ + ' ' +
                self.options.divider + ' ' +
                self.options.answers.replace('data/', '').replace('.csv', ''))

    @property
    def answers(self):
        answers = load_data.get_answers(self.options)
        div = divider.Divider.get_divider(self.options)
        new_column_name = div.column_name
        answers = div.divide(answers, new_column_name)
        self.groupby_column = new_column_name
        return answers


class AnswersByDividerInDay(DivisionInTime):
    date_precision = 2
    date_offset = 11


class AnswersByDividerInDayByMinute(DivisionInTime):
    date_precision = 5
    date_offset = 11


class AnswersByDividerInWeekByHour(DivisionInTime):
    date_precision = 'weekday_and_time'


class AnswersByUserHistogram(Command):
    kind = 'hist'

    def get_data(self):
        answers = load_data.get_answers(self.options)
        grouped = answers.groupby(['user']).count()
        grouped = grouped.reset_index()
        grouped['log_10(count)'] = grouped['id'].map(lambda x: math.log(x, 10))
        grouped = grouped[['log_10(count)']]
        return grouped


class RatingByMap(Command):
    def get_data(self):
        ratings = load_data.get_rating_with_maps(self.options)
        return ratings


class RatingOrderByValue(Command):
    stacked = True

    def get_data(self):
        ratings = load_data.get_rating(self.options)
        grouped = ratings.groupby(['value', 'order']).count()
        grouped = grouped[['user']]
        grouped = grouped.reset_index()
        grouped = grouped.pivot(
            index='value',
            columns='order',
            values='user')
        value_columns = grouped.columns
        grouped = grouped.fillna(0)
        grouped['All'] = 0
        for c in value_columns:
            grouped['All'] += grouped[c]
        for c in value_columns:
            grouped[c] = grouped[c] / grouped['All']
        grouped = grouped[value_columns]
        return grouped


class RatingByOrder(Command):
    stacked = True
    legend_loc = 'center right'

    def get_data(self):
        ratings = load_data.get_rating(self.options)
        grouped = ratings.groupby(['value', 'order']).count()
        grouped = grouped[['user']]
        grouped = grouped.reset_index()
        grouped = grouped.pivot(
            index='order',
            columns='value',
            values='user')
        value_columns = grouped.columns
        grouped = grouped.fillna(0)
        grouped['All'] = 0
        for c in value_columns:
            grouped['All'] += grouped[c]
        for c in value_columns:
            grouped[c] = grouped[c] / grouped['All']
        grouped = grouped[value_columns]
        grouped.rename(columns=RATING_VALUES, inplace=True)
        return grouped


class RatingByLastOrder(Command):
    stacked = True
    legend_loc = 'center right'

    def get_data(self):
        ratings = load_data.get_rating(self.options)
        grouped = ratings.groupby(['value', 'last_order']).count()
        grouped = grouped[['user']]
        grouped = grouped.reset_index()
        grouped = grouped.pivot(
            index='last_order',
            columns='value',
            values='user')
        value_columns = grouped.columns
        grouped = grouped.fillna(0)
        grouped['All'] = 0
        for c in value_columns:
            grouped['All'] += grouped[c]
        for c in value_columns:
            grouped[c] = grouped[c] / grouped['All']
        grouped = grouped[value_columns]
        grouped.rename(columns=RATING_VALUES, inplace=True)
        return grouped


class RatingValueHistogram(Command):
    kind = 'hist'

    def get_data(self):
        ratings = load_data.get_rating(self.options)
        ratings = ratings[['value']]
        return ratings


class RatingOrderHistogram(Command):
    kind = 'hist'

    def get_data(self):
        ratings = load_data.get_rating(self.options)
        ratings = ratings[['order']]
        return ratings


class RatingInTime(InTimeCommand):
    groupby_column = 'value'

    @property
    def answers(self):
        return load_data.get_rating(self.options)


class RatingByDivider(DivisionCommand):
    stacked = True
    adjust_bottom = 0.2

    def get_data(self):
        ratings = load_data.get_rating(self.options)
        ratings = ratings[['user', 'value']]
        div = divider.Divider.get_divider(self.options)
        answers = load_data.get_answers(self.options)
        new_column_name = div.column_name
        answers = div.divide(answers, new_column_name)
        users = answers.drop_duplicates(['user'])
        ratings = pd.merge(
            ratings,
            users,
            left_on=['user'],
            right_on=['user'],
        )
        ratings = ratings[['value', new_column_name, 'user']]
        grouped = ratings.groupby(['value', new_column_name]).count()
        grouped = grouped.reset_index()
        grouped = grouped.pivot(
            index=new_column_name,
            columns='value',
            values='user')
        value_columns = grouped.columns
        grouped = grouped.fillna(0)
        grouped['All'] = 0
        for c in value_columns:
            grouped['All'] += grouped[c]
        for c in value_columns:
            grouped[c] = grouped[c] / grouped['All']
        grouped = grouped[value_columns]
        grouped.rename(columns=RATING_VALUES, inplace=True)
        return grouped


class FeedbackInTime(InTimeCommand):
    groupby_column = 'const'
    date_precision = 7
    absolute_values = True
    kind = 'bar'

    @property
    def answers(self):
        return load_data.get_feedback_data()


class FeedbackByWeekday(InTimeCommand):
    groupby_column = 'const'
    date_precision = 'weekday'
    absolute_values = True
    kind = 'bar'

    @property
    def answers(self):
        return load_data.get_feedback_data()


class FeedbackByType(Command):
    kind = 'pie'
    subplots = True
    legend_bbox = (0.75, -01.0)

    def get_data(self):
        feedback = load_data.get_feedback_data_with_type()
        print len(feedback)
        grouped = feedback.groupby(['type']).count()
        grouped = grouped[['inserted']]
        grouped = grouped.sort('inserted', ascending=False)
        grouped.columns = ['']
        return grouped


class FeedbackWordCountHistogram(Command):
    kind = 'hist'

    def get_data(self):
        feedback = load_data.get_feedback_data()
        feedback['log_10(word_count)'] = feedback['word_count'].map(lambda x: math.log(x, 10))
        feedback = feedback[['log_10(word_count)']]
        return feedback
