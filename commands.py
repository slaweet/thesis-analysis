# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import load_data
import divider
import pandas as pd
import utils
import datetime
import math
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap
import random
import bisect
import seaborn as sns
import os


sns.set_style("whitegrid", {
    'legend.frameon': True,
})


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

AB_VALUES = {
    6: 'Random-Adaptive',
    7: 'Random-Random',
    8: 'Adaptive-Adaptive',
    9: 'Adaptive-Random',
    14: '50%',
    15: '65%',
    16: '80%',
    17: '95%',
}

AB_VALUES_SHORT = {
    6: 'R-A',
    7: 'R-R',
    8: 'A-A',
    9: 'A-R',
    14: '50%',
    15: '65%',
    16: '80%',
    17: '95%',
}


class Command(object):

    @staticmethod
    def name(self):
        return utils.convert_from_cammel_case(self.__name__)

    def __init__(self, options, show_plots=True):
        self.options = options
        self.show_plots = show_plots

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


class PlotCommand(Command):
    kind = 'bar'
    subplots = False
    adjust_bottom = 0.1
    adjust_hspace = None
    adjust_right = 0.95
    legend_alpha = False
    ylim = None
    xlim = None
    legend_loc = None
    legend_bbox = None
    active = True
    rot = None
    stacked = None
    week_cache = {}
    scatter_x = None
    scatter_y = None
    scatter_c = None
    fontsize = 15
    legend = None
    color = None
    marker = None
    subplot_x_dim = None
    figsize = (8, 6)
    colormap = None
    edgecolor = None
    subplots_adjust = None
    subplots_first = 1

    def __init__(self, options, show_plots=True):
        self.options = options
        self.show_plots = show_plots

    def get_plot_params(self, subplot_index):
        if subplot_index > 0:
            self.legend = False
        plot_params = dict(
            kind=self.kind,
            subplots=self.subplots,
            title=self.plot_name() if not self.options.production else '',
            rot=self.rot,
            stacked=self.stacked,
            fontsize=self.fontsize,
            legend=self.legend,
            figsize=self.figsize,
            colormap=self.colormap,
        )
        if self.edgecolor is not None:
            plot_params.update(dict(
                edgecolor=self.edgecolor,
            ))
        if self.marker is not None:
            plot_params.update(dict(
                marker=self.marker,
            ))
        if self.color is not None:
            plot_params.update(dict(
                color=self.color,
            ))
        if self.scatter_c is not None:
            plot_params.update(dict(
                x=self.scatter_x,
                y=self.scatter_y,
                c=self.scatter_c,
                # s=20,
                # marker='+',
            ))
        return plot_params

    def generate_graph(self, data):
        if type(data) is not list:
            data_list = [[data, None]]
        else:
            data_list = data

        fig = plt.gcf()

        for i in range(len(data_list)):
            data = data_list[i][0]
            print len(data)
            print (data.head() if len(data) > 10 else data)
            plot_params = self.get_plot_params(i)
            if self.subplots_adjust is None:
                self.subplots_adjust = dict(
                    bottom=self.adjust_bottom,
                    hspace=self.adjust_hspace,
                    right=self.adjust_right)
            fig.subplots_adjust(**self.subplots_adjust)
            if self.subplot_x_dim is not None:
                ax = fig.add_subplot(math.ceil(len(data_list) / self.subplot_x_dim),
                                     self.subplot_x_dim,
                                     i + self.subplots_first)
            else:
                ax = fig.add_subplot(round(math.sqrt(len(data_list))),
                                     math.ceil(math.sqrt(len(data_list))), i + self.subplots_first)

            if self.ylim is not None:
                ax.set_ylim(self.ylim)

            if self.xlim is not None:
                ax.set_xlim(self.xlim)

            if len(data_list) > 1:
                plot_params.update(dict(
                    title=data.columns.levels[0][0] if hasattr(data.columns, 'levels') else data_list[i][1],
                ))
            plot_params.update(dict(
                ax=ax,
            ))

            data.plot(**plot_params)
            data.to_pickle(self.pickle_name(i))

            if self.legend is False:
                pass
            elif self.legend_loc is not None:
                legend = ax.legend(loc=self.legend_loc, fontsize=self.fontsize)
            elif self.legend_bbox is not None:
                legend = ax.legend(bbox_to_anchor=self.legend_bbox)
            else:
                legend = ax.legend()
            if self.legend_alpha and legend is not None:
                legend.get_frame().set_alpha(0.8)

        plt.savefig(self.file_name())
        if self.options.production:
            plt.savefig(self.file_name().replace('.png', '.svg'))
        if self.show_plots and not self.options.hide_plots:
            plt.show()
        plt.clf()

    def pickle_name(self, i):
        return self.file_name().replace(
            '.png', '-%d.pdy' % i).replace(
            'plot', 'plot_data')

    def file_name(self):
        if self.options.answers == DATA_DIR + 'answers.csv':
            dest_dir = PLOT_DIR
        else:
            dest_dir = PARTIAL_DATA_PLOT_DIR
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        return (dest_dir + utils.convert_from_cammel_case(
            self.plot_name()
        ).replace(' ', '_') + '.png')

    def plot_name(self):
        return (self.__class__.__name__ +
                (' ' + self.options.answers.replace('data/', '').replace(
                    '/', '_').replace(
                    '.csv', '') if not self.options.production else '') +
                (' ' + self.options.context_name if
                    self.options.context_name is not None else '') +
                (' ' + self.options.term_type if
                    self.options.term_type is not None else ''))

    def _execute(self):
        if self.options.use_cached_data:
            data = []
            for i in range(100):
                if not os.path.isfile(self.pickle_name(i)):
                    break
                data.append([pd.read_pickle(self.pickle_name(i)), ''])
        else:
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
        answers[field_name] = answers['time'].map(
            lambda x:
            x[:4] + ' week ' +
            str(self.get_week(x)).zfill(2))
        return answers

    def add_weekday(self, answers, field_name):
        answers[field_name] = answers['time'].map(
            lambda x: str(self.get_weekday(x)))
        return answers

    def add_weekday_and_time(self, answers, field_name):
        weekdays = 'MON THU WED THU FRI SAT SUN'.split()
        for i in range(len(weekdays)):
            weekdays[i] = ' ' * (len(weekdays) - i) + weekdays[i]
        answers[field_name] = answers['time'].map(
            lambda x: str(weekdays[self.get_weekday(x)]) +
            ' ' + x[11:13] + ':00')
        return answers

    def get_data(self):
        pass


class DivisionCommand(PlotCommand):

    def plot_name(self):
        return (self.__class__.__name__ + ' ' +
                self.options.divider +
                (' ' + self.options.answers.replace(
                    'data/', '').replace(
                    '/', '_').replace(
                    '.csv', '') if not self.options.production else ''))


class MnemonicsEffect(PlotCommand):
    active = False
    seen_times = {}

    def increment_seen(self, row):
        user = row['user_id']
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
        # print answers[['user_id', 'seen_times']]
        return answers

    def get_data(self):
        answers = load_data.get_answers_with_flashcards(self.options)
        answers = answers.sort(['place_asked', 'user_id'], ascending=True)
        answers = answers.reset_index()
        answers = self.add_see_times(answers)
        answers = answers[answers['seen_times'] == 1]

        MNEMONICS_ADDED_ON = '2014-12-20 06:23:00'
        mnemonics = load_data.get_mnemonics(self.options)
        answers['after_mnemonics'] = answers['time'] > MNEMONICS_ADDED_ON
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


class FilterLithuania(PlotCommand):
    active = False

    def get_data(self):
        answers = load_data.get_answers(self.options)
        answers = answers[answers.place_asked == 142]
        print answers
        answers.to_csv('data/answers-lithuania.csv', sep=',', encoding='utf-8')
        return answers


class FilterEuropeStates(PlotCommand):
    active = False

    def get_data(self):
        answers = load_data.get_answers_with_flashcards(self.options)
        print len(answers)
        answers = answers[answers.context_name == 'Europe']
        answers = answers[answers.term_type == 'state']
        print len(answers)
        answers_only = load_data.get_answers(self.options)
        print answers.columns, answers_only.columns
        answers = answers[answers_only.columns]
        answers.to_csv('data/answers-europe.csv', sep=',', encoding='utf-8')
        return answers


class PlacesByMap(PlotCommand):
    adjust_bottom = 0.3

    def get_data(self):
        maps = load_data.get_maps(self.options)
        maps = maps.groupby(['context_name']).count()
        maps = maps[['place']]
        maps = maps.sort('place', ascending=False)
        maps.columns = ['Number Of Places']
        return maps


class PlacesByMapAndType(PlotCommand):
    adjust_bottom = 0.3

    def get_data(self):
        maps = load_data.get_maps(self.options)
        maps = maps.groupby(['context_name', 'term_type']).count()
        maps = maps.drop(['context_name', 'term_type', 'map', 'place_name'], 1)
        maps = maps.reset_index()
        maps = maps.pivot(
            index='context_name',
            columns='term_type',
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


class SuccessByMap(PlotCommand):
    def get_data(self):
        answers_with_maps = load_data.get_answers_with_map_grouped(self.options)
        success_rate_by_map = answers_with_maps.mean().sort('correct')
        return success_rate_by_map


class Division(DivisionCommand):
    legend_loc = 'best'
    fontsize = 25
    ylim = [0, 1]
    legend = False

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
        answers_with_maps = load_data.get_answers_with_flashcards(self.options)
        div = divider.Divider.get_divider(self.options)
        new_column_name = div.column_name
        answers_with_maps = div.divide(answers_with_maps, new_column_name)
        answers_with_maps = answers_with_maps.groupby(['context_name', new_column_name])
        answers_by_map = answers_with_maps.count()
        answers_by_map = answers_by_map[['context_id']]
        answers_by_map = answers_by_map.reset_index()
        answers_by_map = answers_by_map.pivot(
            index='context_name',
            columns=new_column_name,
            values='context_id')
        answers_by_map[False] = answers_by_map[False] / answers_by_map[False].sum()
        answers_by_map[True] = answers_by_map[True] / answers_by_map[True].sum()
        answers_by_map['Diff'] = answers_by_map[True] - answers_by_map[False]
        answers_by_map = answers_by_map.sort(False, ascending=False)
        answers_by_map = answers_by_map[:10]
        answers_by_map = answers_by_map.sort('Diff', ascending=False)
        #  answers_by_map.columns = ['Number Of Answers']
        return answers_by_map


class InTimeCommand(PlotCommand):
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
            answers = load_data.get_answers_with_flashcards(self.options)
        if self.date_precision == 'week':
            answers = self.add_week(answers, 'date')
        elif self.date_precision == 'weekday':
            answers = self.add_weekday(answers, 'date')
        elif self.date_precision == 'weekday_and_time':
            answers = self.add_weekday_and_time(answers, 'date')
        else:
            answers['date'] = answers['time'].map(
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
    groupby_column = 'context_name'
    date_precision = 'week'


class AnswersByMapInDay(AnswersByMapInTime):
    date_precision = 2
    date_offset = 11


class AnswersByMapInDayAbsolute(AnswersByMapInDay):
    absolute_values = True


class UsersByMapInTime(AnswersByMapInTime):
    drop_duplicate = 'user_id'


class AnswersByLangInTime(InTimeCommand):
    legend_loc = 'lower left'
    groupby_column = 'metainfo_id'
    columns_rename = LANGUAGES


class AnswersByLangInDay(AnswersByLangInTime):
    date_precision = 2
    date_offset = 11


class AnswersInDayAbsolute(AnswersByLangInTime):
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
    adjust_bottom = 0.2


class UsersInDayAbsolute(AnswersInDayAbsolute):
    columns_rename = {'0': 'Number of users'}
    drop_duplicate = 'user_id'


class UsersInDayByMinuteAbsolute(AnswersInDayAbsolute):
    columns_rename = {'0': 'Number of users'}
    drop_duplicate = 'user_id'
    date_precision = 5


class AnswersInTimeAbsolute(AnswersByLangInTime):
    result_columns = None
    columns_rename = {'0': 'Number of answers'}
    legend_loc = 'lower right'
    date_precision = 7
    absolute_values = True
    adjust_bottom = 0.1


class UsersInTimeAbsolute(AnswersInTimeAbsolute):
    columns_rename = {'0': 'Number of users'}
    drop_duplicate = 'user_id'


class AnswersByPlaceTypeInTime(InTimeCommand):
    legend_loc = 'lower left'
    groupby_column = 'term_type'
    columns_rename = PLACE_TYPES
    result_columns = [1, 2, 5, 7, 13, 14]
    date_precision = 'week'


class AnswersByNumberOfOptionsInTime(InTimeCommand):
    legend_loc = 'lower right'
    groupby_column = 'guess'


class CorrectRateInTime(InTimeCommand):
    legend_loc = 'upper right'
    groupby_column = 'correct'
    date_precision = 9


class DividerSimilarity(PlotCommand):
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


class DividerCorrelation(PlotCommand):
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
                (self.options.answers.replace(
                    'data/', '').replace(
                    '/', '_').replace(
                    '.csv', '') if not self.options.production else ''))

    @property
    def answers(self):
        answers = load_data.get_answers(self.options)
        div = divider.Divider.get_divider(self.options)
        new_column_name = div.column_name
        answers = div.divide(answers, new_column_name)
        self.groupby_column = new_column_name
        return answers


class ResponseTimeByDivider(Command):
    def _execute(self):
        answers = load_data.get_answers(self.options, strip_times=True)
        div = divider.Divider.get_divider(self.options)
        new_column_name = div.column_name
        answers = div.divide(answers, new_column_name)
        answers['response_time'] = answers['response_time'].apply(
            lambda x: math.log(x, 2))
        cat1 = answers[answers[div.column_name]]['response_time']
        cat2 = answers[~answers[div.column_name]]['response_time']
        print cat1.describe()
        print cat2.describe()
        print stats.ranksums(cat1, cat2)
        """
        answers = answers[[new_column_name, 'response_time', 'user_id']]
        grouped = answers.groupby(['user_id', new_column_name]).median()
        # grouped = grouped.reset_index()
        return grouped
        """


class ResponseTimeVsSkill(PlotCommand):
    kind = 'scatter'
    scatter_x = 'response_time'
    scatter_y = 'skill'
    scatter_c = 'log_10(count)'

    def get_data(self):
        answers = load_data.get_answers(
            self.options, strip_times=True,
            strip_less_than_10=True)
        div = divider.Divider.get_divider(self.options)
        # self.scatter_c = div.column_name
        answers = div.divide(answers, div.column_name)
        grouped = answers.groupby(['user_id']).median()
        response_times = grouped.reset_index()
        skills = load_data.get_prior_skills(self.options)
        skills.rename(columns={'value': 'skill'}, inplace=True)
        answers_with_skills = pd.merge(
            response_times,
            skills,
            left_on=['user_id'],
            right_on=['user_id'],
        )
        counts = answers.groupby(['user_id']).count().reset_index()
        counts['log_10(count)'] = counts['id'].map(lambda x: math.log(x, 10))
        counts = counts[['user_id', 'log_10(count)']]
        answers_with_skills = pd.merge(
            answers_with_skills,
            counts,
            left_on=['user_id'],
            right_on=['user_id'],
        )
        answers_with_skills = answers_with_skills[
            [self.scatter_x, self.scatter_y, self.scatter_c]]
        # print answers_with_skills.describe()
        corr = answers_with_skills.corr()
        print corr
        return answers_with_skills


class AnswersByDividerInDay(DivisionInTime):
    date_precision = 2
    date_offset = 11


class AnswersByDividerInDayByMinute(DivisionInTime):
    date_precision = 5
    date_offset = 11


class AnswersByDividerInWeekByHour(DivisionInTime):
    date_precision = 'weekday_and_time'


class AnswersByUserHistogram(PlotCommand):
    kind = 'hist'

    def get_data(self):
        answers = load_data.get_answers(self.options)
        grouped = answers.groupby(['user_id']).count()
        grouped = grouped.reset_index()
        grouped['log_10(count)'] = grouped['id'].map(lambda x: math.log(x, 10))
        grouped = grouped[['log_10(count)']]
        return grouped


class AnswerPlacesHistogram(PlotCommand):
    adjust_bottom = 0.2

    def get_data(self):
        answers = load_data.get_answers_with_flashcards(self.options)
        answers = answers[answers['context_name'] == 'Europe']
        answers = answers[answers['term_type'] == 1]
        grouped = answers.groupby(['place_name']).count()
        # grouped = grouped.reset_index()
        grouped = grouped.sort('id', ascending=False)
        grouped = grouped[['id']]
        return grouped


class AnswerPlacesByUserHistogram(PlotCommand):
    adjust_bottom = 0.2

    def get_data(self):
        answers = load_data.get_answers_with_flashcards(self.options)
        answers = answers[answers['context_name'] == 'Europe']
        answers = answers[answers['term_type'] == 1]
        grouped = answers.groupby(['place_name', 'user_id']).count()
        # grouped = grouped.reset_index()
        grouped = grouped.sort('id', ascending=False)
        grouped = grouped[['id']]
        return grouped


class RatingByMap(PlotCommand):
    subplots_adjust = dict(
        bottom=0.5,
    )
    ylim = [0, 1]
    subplot_x_dim = 4
    figsize = (16, 6)

    def make_relative(self, grouped):
        value_columns = grouped.columns
        grouped = grouped.fillna(0)
        grouped['All'] = 0
        for c in value_columns:
            grouped['All'] += grouped[c]
        for c in value_columns:
            grouped[c] = grouped[c] / grouped['All']
        grouped = grouped[value_columns]
        return grouped

    def get_data(self):
        all_ratings = load_data.get_rating_with_maps(self.options)
        data = []
        ratings = all_ratings
        ratings['Context'] = ratings['context_name'] + ' - ' + ratings['term_type']
        ratings = ratings[['Context',  'user_id', 'value', 'experiment_setup_id']]
        ratings = ratings.groupby(['Context', 'value', 'experiment_setup_id']).count()
        all_ratings = ratings.reset_index()
        ratings = ratings.sort('user_id', ascending=False)
        ratings = ratings[:10]
        for i in all_ratings['Context'].tolist():
            print 'i', i
            ratings = all_ratings[all_ratings['Context'] == i]
            print 'tr', ratings
            ratings = ratings[['user_id', 'value', 'experiment_setup_id']]
            ratings = ratings.pivot(
                index='experiment_setup_id',
                columns='value',
                values='user_id')
            ratings['all'] = ratings[1] + ratings[2] + ratings[3]
            ratings = ratings.sort('all', ascending=False)
            ratings = ratings[:10]
            ratings.drop('all', axis=1, inplace=True)
            # ratings.rename(columns=RATING_VALUES, inplace=True)
            ratings = self.make_relative(ratings)
            data.append([ratings, i])
        return data


class RatingByAnswerCount(PlotCommand):
    ylim = [0, 1]
    legend_alpha = True
    answer_counts = [30, 70, 120, 200, 300, 500]

    def make_relative(self, grouped):
        value_columns = grouped.columns
        grouped = grouped.fillna(0)
        grouped['All'] = 0
        for c in value_columns:
            grouped['All'] += grouped[c]
        for c in value_columns:
            grouped[c] = grouped[c] / grouped['All']
        grouped = grouped[value_columns]
        return grouped

    def get_data(self):
        all_ratings = load_data.get_rating_with_maps(self.options)
        answers = load_data.get_answers_with_flashcards_and_orders(self.options)
        answers = answers.groupby('user_id').count()[['answer_order']]
        answers['answer_order'] = answers['answer_order'].map(
            lambda x: self.answer_counts[bisect.bisect_left(self.answer_counts, x) - 1])
        answers.rename(columns={'answer_order': '(Answer count)'}, inplace=True)
        answers = answers.reset_index()
        print answers
        all_ratings = pd.merge(
            all_ratings,
            answers,
            left_on=['user_id'],
            right_on=['user_id'],
        )
        data = []
        for i in AB_VALUES:
            ratings = all_ratings[all_ratings['experiment_setup_id'] == i]
            ratings = ratings[['(Answer count)',  'user_id', 'value']]
            ratings = ratings.groupby(['(Answer count)', 'value']).count()
            ratings = ratings.reset_index()
            ratings = ratings.pivot(
                index='(Answer count)',
                columns='value',
                values='user_id')
            ratings.rename(columns=RATING_VALUES, inplace=True)
            ratings = self.make_relative(ratings)
            data.append([ratings, AB_VALUES[i]])
        return data


class ResponseTimeByPrevious(PlotCommand):
    # kind = 'hist'

    def add_prev(self, answers, key):
        answers = answers.sort(['user_id', 'time'], ascending=False)
        prev_correct = []
        grouped = answers.groupby('user_id')
        for i, g in grouped:
            prev = None
            for j, row in g[key].iteritems():
                if prev is not None:
                    prev_correct.append(row)
                else:
                    prev_correct.append(-1)
                prev = row
        answers['prev_' + key] = pd.Series(prev_correct, index=answers.index)
        return answers

    def get_data(self):
        answers = load_data.get_answers_with_flashcards_and_orders(self.options)
        answers = self.add_prev(answers, 'correct')
        answers = self.add_prev(answers, 'direction')
        answers = answers[answers['prev_correct']!=-1]
        answers = answers[answers['response_time'] < 30000]
        answers = answers[answers['response_time'] > 0]
        # answers = answers[answers['correct']==True]
        answers['same_direction'] = answers['direction'] == answers['prev_direction']
        answers = answers[['prev_correct', 'same_direction', 'response_time']]
        times = answers.groupby(['prev_correct', 'same_direction']).median()
        return times


class RatingByContextSize(PlotCommand):
    kind = 'area'
    stacked = True
    adjust_bottom = 0.1
    ylim = [0, 1]
    legend_loc = 'center'
    subplots_adjust = dict(
        hspace=0.4,
    )
    legend_alpha=True

    def make_relative(self, grouped):
        value_columns = grouped.columns
        grouped = grouped.fillna(0)
        grouped['All'] = 0
        for c in value_columns:
            grouped['All'] += grouped[c]
        for c in value_columns:
            grouped[c] = grouped[c] / grouped['All']
        grouped = grouped[value_columns]
        return grouped

    def get_data(self):
        ratings = load_data.get_rating_with_maps(self.options)
        res2 = []
        res = None
        ab_values = sorted(ratings['experiment_setup_id'].unique().tolist())
        for i in ab_values:
            ratings_ab = ratings[ratings['experiment_setup_id'] == i]
            grouped = ratings_ab.groupby(['value', 'context_size']).count()
            grouped = grouped[['inserted']]
            grouped = grouped.reset_index()
            grouped = grouped.pivot(
                index='context_size',
                columns='value',
                values='inserted')
            grouped = self.make_relative(grouped)
            grouped.rename(columns=RATING_VALUES, inplace=True)
            if res is None:
                res = grouped
            else:
                res = res.join(grouped, how='right', lsuffix='_x')
            res2.append([grouped, AB_VALUES[i]])
        return res2


class FirstAnswerUnanswered(InTimeCommand):
    kind = 'line'
    stacked = False
    legend_loc = 'lower right'
    ylim = None

    def get_data(self):
        answers = load_data.get_answers_with_flashcards_and_orders(self.options)
        answers['is_not_answered'] = answers['item_answered_id'].map(lambda x: math.isnan(x))
        answers['is_first'] = answers['answer_order'].map(lambda x: x == 1)
        answers = self.add_week(answers, 'date')
        answers = answers[['date', 'is_not_answered', 'is_first']]
        grouped = answers.groupby(['date', 'is_first']).mean()
        grouped = grouped.reset_index()
        grouped = grouped.pivot(
            index='date',
            columns='is_first',
            values='is_not_answered')
        return grouped


class SecondAnswer(PlotCommand):
    kind = 'line'
    legend_loc = 'lower right'

    def get_data(self):
        answers = load_data.get_answers_with_flashcards_and_orders(self.options)
        answers = answers[answers['answer_order'] == 3]
        answers = answers[answers['context_id'] != 17]
        answers = answers[answers['metainfo_id'] == 1]
        # grouped = answers.groupby(['user_id']).count()
        # grouped = grouped[['id']]
        # grouped = grouped.reset_index()
        # grouped = grouped[grouped['id'] == 2]
        grouped = answers
        return grouped


class AnswerOrder(PlotCommand):
    kind = 'line'
    legend_loc = 'lower right'


class ErrorRateByAnswerOrder(AnswerOrder):
    legend_loc = 'upper right'
    marker = '+'

    def get_data(self):
        answers = load_data.get_answers_with_flashcards_and_orders(self.options)
        answers = answers[answers['answer_order'] <= 100]
        grouped = answers.groupby(['answer_order', 'experiment_setup_id']).mean()
        grouped['error_rate'] = 1 - grouped['correct']
        grouped = grouped[['error_rate']]
        grouped = grouped.reset_index()
        grouped = grouped.pivot(
            index='answer_order',
            columns='experiment_setup_id',
            values='error_rate')
        grouped.columns = [AB_VALUES[i] for i in grouped.columns]
        grouped = grouped.reindex_axis(sorted(grouped.columns), axis=1)
        return grouped


class GuessByAnswerOrder(AnswerOrder):
    marker = None

    def get_data(self):
        answers = load_data.get_answers_with_flashcards_and_orders(self.options)
        answers = answers[answers['answer_order'] <= 60]
        grouped = answers.groupby(['answer_order', 'experiment_setup_id']).mean()
        grouped = grouped[['guess']]
        grouped = grouped.reset_index()
        grouped = grouped.pivot(
            index='answer_order',
            columns='experiment_setup_id',
            values='guess')
        grouped.columns = [AB_VALUES[i] for i in grouped.columns]
        grouped = grouped.reindex_axis(sorted(grouped.columns), axis=1)
        return grouped


class UnansweredByAb(AnswerOrder):
    legend_loc = 'upper right'

    def get_data(self):
        answers = load_data.get_answers_with_flashcards_and_orders(self.options)
        answers['is_not_answered'] = answers['item_answered_id'].map(lambda x: math.isnan(x))
        answers = answers[answers['answer_order'] <= 60]
        grouped = answers.groupby(['answer_order', 'experiment_setup_id']).mean()
        answers = answers[['is_not_answered']]
        grouped = grouped.reset_index()
        grouped = grouped.pivot(
            index='answer_order',
            columns='experiment_setup_id',
            values='is_not_answered')
        grouped.columns = [AB_VALUES[i] for i in grouped.columns]
        grouped = grouped.reindex_axis(sorted(grouped.columns), axis=1)
        return grouped


class LearningCurves(AnswerOrder):
    max_answer_order = 70

    def get_curve_data(self, answers):
        answers = answers[answers['metainfo_id'] == 1]
        answers = answers[answers['answer_order'].isin(range(1, self.max_answer_order, 10))]
        grouped = answers.groupby(['answer_order', 'experiment_setup_id']).mean()
        grouped = grouped[['correct']]
        grouped = grouped.reset_index()
        grouped = grouped.pivot(
            index='answer_order',
            columns='experiment_setup_id',
            values='correct')
        grouped.columns = [AB_VALUES[i] for i in grouped.columns]
        grouped = grouped.reindex_axis(sorted(grouped.columns), axis=1)
        return grouped

    def get_data(self):
        answers = load_data.get_answers_with_flashcards_and_context_orders(self.options)
        curve_data = self.get_curve_data(answers)
        return curve_data


class LearningCurvesWithTotalCount(AnswerOrder):
    max_answer_order = 190
    legend_alpha = True
    legend_loc = None
    legend_bbox = (-0.2, 1.2)
    subplots_first = 2
    marker = 'o'
    subplots_adjust = dict(
        wspace=0.7,
        hspace=0.7,
    )
    ylim = [0.35, 0.9]
    cumulative = False

    def get_curve_data(self, answers):
        answers = answers[answers['metainfo_id'] == 1]
        answers = answers[answers['answer_order'].isin(range(1, self.max_answer_order, 10))]
        data = []
        for i in range(1, 9):
            filtered = answers.groupby('user_id').count()['id'].reset_index()
            if self.cumulative:
                filtered = filtered[filtered['id'] >= i]
            else:
                filtered = filtered[filtered['id'] == i]
            users = filtered['user_id'].tolist()

            filtered = answers[answers['user_id'].isin(users)]
            filtered = filtered[filtered['answer_order'] <= i * 10]
            grouped = filtered.groupby(['answer_order', 'experiment_setup_id']).mean()

            grouped = grouped[['correct']]
            grouped = grouped.reset_index()
            grouped = grouped.pivot(
                index='answer_order',
                columns='experiment_setup_id',
                values='correct')
            grouped.columns = [AB_VALUES[j] for j in grouped.columns]
            grouped = grouped.reindex_axis(sorted(grouped.columns), axis=1)
            data.append([grouped, str(i) + '; user_count: ' + str(len(users))])
        return data

    def get_data(self):
        answers = load_data.get_answers_with_flashcards_and_context_orders(self.options)
        curve_data = self.get_curve_data(answers)
        return curve_data


class LearningCurvesWithTotalCountCumulative(LearningCurvesWithTotalCount):
    cumulative = True


class LearningCurvesByRating(LearningCurves):
    legend = False
    ylim = [0.25, 1]
    max_answer_order = 50

    def get_data(self):
        ratings = load_data.get_rating(self.options)
        ratings = ratings.groupby(['user_id']).mean()
        ratings = ratings[['value']]
        ratings = ratings.reset_index()

        answers = load_data.get_answers_with_flashcards_and_context_orders(self.options)
        answers = pd.merge(
            ratings,
            answers,
            on=['user_id'],
        )

        res = []
        all_answers = answers
        for i in range(0, 8):
            answers = all_answers[all_answers['value'] == i / 2.0]
            curve_data = self.get_curve_data(answers)
            if len(curve_data) > 0:
                res.append(curve_data)
        print len(res)
        return res


class LearningCurvesByDivider(LearningCurves):
    legend = False
    ylim = [0.4, 0.8]
    max_answer_order = 50
    subplot_x_dim = 2
    figsize = (9, 13)

    def get_data(self):
        res = []
        self.options.divider = 'all'
        divs = divider.Divider.get_divider(self.options)
        all_answers = load_data.get_answers_with_flashcards_and_context_orders(self.options)

        for i in divs:
            div = divs[i](self.options)
            print div.column_name
            new_column_name = div.column_name

            answers = div.divide(all_answers, new_column_name)
            true_answers = answers[answers[new_column_name]]
            curve_data = self.get_curve_data(true_answers)
            if len(curve_data) > 0:
                res.append(curve_data)

            false_answers = answers[~answers[new_column_name]]
            curve_data = self.get_curve_data(false_answers)
            if len(curve_data) > 0:
                res.append(curve_data)
        return res


class ErrorRateByAnswerOrderByContext(AnswerOrder):
    adjust_bottom = 0.2
    legend_alpha = True
    legend_loc = 'upper right'
    subplots_adjust = dict(
        wspace=0.5,
        hspace=0.3,
    )
    ylim = [0, 0.6]
    marker = '+'
    figsize = (20, 15)

    def get_data(self):
        answers = load_data.get_answers_with_flashcards_and_context_orders(self.options)
        answers['Context'] = answers['context_name'] + ' - ' + answers['term_type']
        top_contexts = answers.groupby('Context').count()[['id']].sort(
            ['id'], ascending=[False]).head(12).reset_index()['Context'].tolist()
        answers = answers[answers['answer_order'] <= 100]
        data = []
        for context in top_contexts:
            answers_on_context = answers[answers['Context'] == context]
            grouped = answers_on_context.groupby(['answer_order', 'experiment_setup_id']).mean()
            grouped['error_rate'] = 1 - grouped['correct']
            grouped = grouped[['error_rate']]
            grouped = grouped.reset_index()
            grouped = grouped.pivot(
                index='answer_order',
                columns='experiment_setup_id',
                values='error_rate')
            grouped.columns = [AB_VALUES_SHORT[i] for i in grouped.columns]
            grouped = grouped.reindex_axis(sorted(grouped.columns), axis=1)
            data.append([grouped, context])
        return data


class ErrorRateBySetOrderByContext(AnswerOrder):
    adjust_bottom = 0.2
    legend_alpha = True
    legend_loc = 'upper right'
    subplots_adjust = dict(
        wspace=0.5,
        hspace=0.3,
    )
    ylim = [0, 0.6]
    marker = '+'
    figsize = (20, 15)

    def get_data(self):
        answers = load_data.get_answers_with_flashcards_and_context_orders(self.options)
        answers = answers[answers['metainfo_id'] != 1]
        answers['set_order'] = answers['answer_order'].apply(lambda x: x / 10)
        answers['Context'] = answers['context_name'] + ' - ' + answers['term_type']
        top_contexts = answers.groupby('Context').count()[['id']].sort(
            ['id'], ascending=[False]).head(12).reset_index()['Context'].tolist()
        answers = answers[answers['set_order'] <= 10]
        data = []
        for context in top_contexts:
            answers_on_context = answers[answers['Context'] == context]
            grouped = answers_on_context.groupby(['set_order', 'experiment_setup_id']).mean()
            grouped['error_rate'] = 1 - grouped['correct']
            grouped = grouped[['error_rate']]
            grouped = grouped.reset_index()
            grouped = grouped.pivot(
                index='set_order',
                columns='experiment_setup_id',
                values='error_rate')
            grouped.columns = [AB_VALUES_SHORT[i] for i in grouped.columns]
            grouped = grouped.reindex_axis(sorted(grouped.columns), axis=1)
            data.append([grouped, context])
        return data


class ErrorRateByAnswerOrderOnContext(AnswerOrder):
    legend_loc = 'upper right'
    marker = '+'
    adjust_bottom = 0.2

    def get_data(self):
        answers = load_data.get_answers_with_flashcards_and_context_orders(self.options)
        answers = answers[answers['answer_order'] <= 100]
        grouped = answers.groupby(['answer_order', 'experiment_setup_id']).mean()
        grouped['error_rate'] = 1 - grouped['correct']
        grouped = grouped[['error_rate']]
        grouped = grouped.reset_index()
        grouped = grouped.pivot(
            index='answer_order',
            columns='experiment_setup_id',
            values='error_rate')
        grouped.columns = [AB_VALUES_SHORT[i] for i in grouped.columns]
        grouped = grouped.reindex_axis(sorted(grouped.columns), axis=1)
        return grouped


class ErrorRateBySetOrderOnContext(AnswerOrder):
    legend_loc = 'upper right'
    marker = '+'
    adjust_bottom = 0.2

    def get_data(self):
        answers = load_data.get_answers_with_flashcards_and_context_orders(self.options)
        answers = answers[answers['metainfo_id'] != 1]
        answers['set_order'] = answers['answer_order'].apply(lambda x: x / 10)
        answers = answers[answers['set_order'] <= 10]
        grouped = answers.groupby(['set_order', 'experiment_setup_id']).mean()
        grouped['error_rate'] = 1 - grouped['correct']
        grouped = grouped[['error_rate']]
        grouped = grouped.reset_index()
        grouped = grouped.pivot(
            index='set_order',
            columns='experiment_setup_id',
            values='error_rate')
        grouped.columns = [AB_VALUES_SHORT[i] for i in grouped.columns]
        grouped = grouped.reindex_axis(sorted(grouped.columns), axis=1)
        return grouped


class IsNotAnsweredByAnswerOrderOnContext(AnswerOrder):
    legend_loc = 'upper right'

    def get_data(self):
        answers = load_data.get_answers_with_flashcards_and_context_orders(self.options)
        answers = answers[answers['answer_order'] <= 60]
        answers['is_not_answered'] = answers['item_answered_id'].map(lambda x: math.isnan(x))
        grouped = answers.groupby(['answer_order', 'experiment_setup_id']).mean()
        grouped = grouped[['is_not_answered']]
        grouped = grouped.reset_index()
        grouped = grouped.pivot(
            index='answer_order',
            columns='experiment_setup_id',
            values='is_not_answered')
        grouped.columns = [AB_VALUES[i] for i in grouped.columns]
        grouped = grouped.reindex_axis(sorted(grouped.columns), axis=1)
        return grouped


class AnswerCountByOrder(AnswerOrder):
    kind = 'line'
    legend_loc = 'upper right'

    def get_data(self):
        answers = load_data.get_answers_with_flashcards_and_orders(self.options)
        answers = answers[answers['answer_order'] <= 100]
        grouped = answers.groupby(['answer_order', 'experiment_setup_id']).count()
        grouped = grouped[['correct']]
        grouped = grouped.reset_index()
        grouped = grouped.pivot(
            index='answer_order',
            columns='experiment_setup_id',
            values='correct')
        grouped.columns = [AB_VALUES[i] for i in grouped.columns]
        grouped = grouped.reindex_axis(sorted(grouped.columns), axis=1)
        return grouped


class AnswerCountDiffByOrder(AnswerOrder):
    kind = 'line'
    legend_loc = 'upper right'
    marker = 'o'

    def get_data(self):
        answers = load_data.get_answers_with_flashcards_and_orders(self.options)
        answers = answers[answers['answer_order'] <= 100]
        grouped = answers.groupby(['answer_order', 'experiment_setup_id']).count()
        grouped = grouped[['correct']]
        grouped = grouped.reset_index()
        grouped = grouped.pivot(
            index='answer_order',
            columns='experiment_setup_id',
            values='correct')
        grouped.columns = [AB_VALUES[i] for i in grouped.columns]
        grouped = grouped.reindex_axis(sorted(grouped.columns), axis=1)
        for c in grouped.columns:
            grouped[c] = 1 - grouped[c] * 1.0 / grouped[c].shift(1)
        return grouped


class UserCurve(PlotCommand):
    legend_loc = 'upper right'
    kind = 'line'
    legend_alpha = True
    figsize = (20, 15)
    subplot_x_dim = 6
    adjust_bottom = 0
    adjust_hspace = 0.4

    def get_data(self):
        answers = load_data.get_answers_with_flashcards_and_orders(self.options)

        grouped = answers.groupby('user_id').count()
        grouped = grouped.reset_index()
        more60 = grouped[grouped['time'] >= 60]['user_id']
        answers = answers[answers['user_id'].isin(more60)]
        users = []
        ab_values = sorted(answers['experiment_setup_id'].unique().tolist())
        for i in ab_values:
            answers_ab = answers[answers['experiment_setup_id'] == i]
            users = users + answers_ab['user_id'].unique()[:self.subplot_x_dim * 2].tolist()
        all_answers = answers
        data = []
        for user in users:
            answers = all_answers[all_answers['user_id'] == user]
            answers = answers[answers['answer_order'] <= 60]
            answers.set_index('answer_order', inplace=True)
            answers['context_change'] = answers['context_id'] != answers['context_id'].shift(1)
            answers['session_start'] = answers['session_id'] != answers['session_id'].shift(1)
            answers['response_time_log10'] = answers['response_time'].apply(lambda x: math.log(x, 10))
            answers['rolling_success'] = sum([answers['correct'].shift(i) for i in range(10)]) / 10.0
            answers = answers[[
                # 'correct',
                'rolling_success',
                'guess',
                # 'metainfo_id',
                # 'response_time_log10',
                'context_change',
                'session_start',
            ]]
            data.append([answers, 'User: ' + str(user)])
        return data


class UsageScatter(PlotCommand):
    kind = 'scatter'
    scatter_x = 'answer_order'
    scatter_y = 'user'
    scatter_c = 'correct - guess'
    figsize = (20, 30)
    colormap = LinearSegmentedColormap.from_list('my', ['red', 'orange', 'green'])
    edgecolor = 'none'
    marker = '.'
    xlim = [0, 201]
    ylim = [0, 401]
    random_users = False
    sort_by_length = False
    subplots_adjust = dict(
        hspace=0.05,
        wspace=0.1,
        top=0.95,
        bottom=0.05,
        right=0.95,
        left=0.05,
    )

    def edit_answers(self, answers):
        answers['correct - guess'] = (answers['correct'] + 0 - answers['guess']).apply(lambda x: max(0, x))
        return answers

    def get_data(self):
        all_answers = load_data.get_answers_with_flashcards_and_orders(self.options)
        data = []
        ab_values = sorted(all_answers['experiment_setup_id'].unique().tolist())
        for i in ab_values:
            answers_ab = all_answers[all_answers['experiment_setup_id'] == i]
            if self.random_users:
                users = random.sample(answers_ab['user_id'].unique(), self.ylim[1])
            else:
                users = answers_ab['user_id'].unique()[:self.ylim[1]].tolist()
            answers = answers_ab[answers_ab['user_id'].isin(users)]
            if self.sort_by_length:
                users_dict = dict(zip(zip(*sorted(answers.groupby(['user_id']).apply(len).to_dict().items(), key=lambda x: x[1]))[0], range(len(users))))
            else:
                users_dict = dict([(users[j], j) for j in range(len(users))])
            # print AB_VALUES[i], users_dict
            answers[self.scatter_y] = answers['user_id'].apply(lambda x: users_dict[x])
            answers = answers[answers['answer_order'] < self.xlim[1]]
            answers['context_change'] = answers['context_id'] != answers['context_id'].shift(1)
            answers = self.edit_answers(answers)
            answers = answers[[
                self.scatter_x,
                self.scatter_y,
                self.scatter_c,
            ]]
            data.append([answers, AB_VALUES[i]])
        return data


class UsageScatterSorted(UsageScatter):
    random_users = True
    sort_by_length = True


class UsageScatterResponseTime(UsageScatterSorted):
    scatter_c = 'response_time_log10'
    colormap = LinearSegmentedColormap.from_list('black', ['white', 'black'])

    def edit_answers(self, answers):
        answers['response_time_log10'] = answers['response_time'].apply(lambda x: math.log(min(x, 30000), 10))
        return answers


class UsageScatterMetainfo(UsageScatterSorted):
    scatter_c = 'metainfo_id'
    colormap = LinearSegmentedColormap.from_list('black', ['#cccccc', 'black'])


class UsageScatterGuess(UsageScatterSorted):
    scatter_c = 'guess'
    colormap = LinearSegmentedColormap.from_list('black', ['#cccccc', 'black'])


class UsageScatterSessionStart(UsageScatterSorted):
    scatter_c = 'session_start'
    colormap = LinearSegmentedColormap.from_list('black', ['#cccccc', 'black'])

    def edit_answers(self, answers):
        answers['session_start'] = answers['session_id'] != answers['session_id'].shift(1)
        return answers


class UsageScatterContext(UsageScatter):
    random_users = True
    scatter_c = 'context_id'
    colormap = 'Set1'


class ResponseTimeByAnswerOrder(PlotCommand):
    kind = 'line'
    legend_loc = 'upper right'

    def get_data(self):
        answers = load_data.get_answers_with_flashcards_and_orders(self.options)
        answers = answers[answers['answer_order'] <= 60]
        grouped = answers.groupby(['answer_order']).median()
        grouped = grouped.reset_index()
        grouped = grouped[['response_time']]
        return grouped


class ResponseTimeByAnswerOrderOnContext(PlotCommand):
    kind = 'line'
    legend_loc = 'upper right'

    def get_data(self):
        answers = load_data.get_answers_with_flashcards_and_context_orders(self.options)
        answers = answers[answers['answer_order'] <= 60]
        grouped = answers.groupby(['answer_order']).median()
        grouped = grouped.reset_index()
        grouped = grouped[['response_time']]
        return grouped


class ResponseTimeByGuessAb(PlotCommand):
    kind = 'bar'
    ylim = (0, 7000)
    legend_loc = 'lower right'
    subplots_adjust = dict(
        hspace=0.7,
        wspace=0.3,
        bottom=0.2,
    )

    def get_data(self):
        all_answers = load_data.get_answers(self.options)
        ab_values = sorted(all_answers['experiment_setup_id'].unique().tolist())
        data = []
        for i in ab_values:
            answers = all_answers[all_answers['experiment_setup_id'] == i]
            answers['number of options'] = 1 / answers['guess']
            grouped = answers.groupby(['number of options', 'direction']).median()
            grouped = grouped[['response_time']]
            data.append([grouped,  AB_VALUES[i]])
        return data


class ResponseTimeByAnswerOrderGuess(PlotCommand):
    kind = 'line'
    legend_loc = 'upper right'

    def get_answers(self):
        answers = load_data.get_answers_with_flashcards_and_orders(self.options)
        return answers

    def get_data(self):
        answers = self.get_answers()
        answers = answers[answers['answer_order'] <= 60]
        grouped = answers.groupby(['answer_order', 'guess']).median()
        grouped = grouped[['response_time']]
        grouped = grouped.reset_index()
        grouped = grouped.pivot(
            index='answer_order',
            columns='guess',
            values='response_time')
        grouped = grouped.reindex_axis(sorted(grouped.columns), axis=1)
        return grouped


class ResponseTimeByAnswerOrderAb(PlotCommand):
    kind = 'line'
    legend_loc = 'upper right'

    def get_answers(self):
        answers = load_data.get_answers_with_flashcards_and_orders(self.options)
        return answers

    def get_data(self):
        answers = self.get_answers()
        answers = answers[answers['answer_order'] <= 60]
        grouped = answers.groupby(['answer_order', 'experiment_setup_id']).median()
        grouped = grouped[['response_time']]
        grouped = grouped.reset_index()
        grouped = grouped.pivot(
            index='answer_order',
            columns='experiment_setup_id',
            values='response_time')
        grouped.columns = [AB_VALUES[i] for i in grouped.columns]
        grouped = grouped.reindex_axis(sorted(grouped.columns), axis=1)
        return grouped


class ResponseTimeByAnswerOrderOnContextAb(ResponseTimeByAnswerOrderAb):

    def get_answers(self):
        answers = load_data.get_answers_with_flashcards_and_context_orders(self.options)
        return answers


class ResponseTimeByAnswerOrderAbCorrect(PlotCommand):
    kind = 'line'
    legend_loc = 'upper right'
    figsize = (16, 6)
    subplot_column = 'correct'
    legend_alpha = True
    ylim = (3000, 9000)

    def get_answers(self):
        answers = load_data.get_answers_with_flashcards_and_context_orders(self.options)
        return answers

    def get_data(self):
        all_answers = self.get_answers()
        data = []
        possible_values = sorted(all_answers[self.subplot_column].unique().tolist())
        for i in possible_values:
            answers = all_answers[all_answers[self.subplot_column] == i]
            answers = answers[answers['answer_order'] <= 60]
            grouped = answers.groupby(['answer_order', 'experiment_setup_id']).median()
            grouped = grouped[['response_time']]
            grouped = grouped.reset_index()
            grouped = grouped.pivot(
                index='answer_order',
                columns='experiment_setup_id',
                values='response_time')
            grouped.columns = [AB_VALUES[j] for j in grouped.columns]
            grouped = grouped.reindex_axis(sorted(grouped.columns), axis=1)
            data.append([grouped, self.subplot_column + ': ' + str(i)])
        return data


class ResponseTimeByAnswerOrderAbGuess(ResponseTimeByAnswerOrderAbCorrect):
    subplot_column = 'guess'
    figsize = (8, 6)
    subplot_x_dim = 2


class SuccessByContextSize(PlotCommand):
    kind = 'area'

    def get_data(self):
        answers = load_data.get_answers_with_flashcards(self.options)
        # answers = answers[answers['metainfo_id'] != 1]
        grouped = answers.groupby(['context_size']).mean()
        grouped = grouped[['correct']]
        return grouped


class RatingOrderByValue(PlotCommand):
    stacked = True

    def get_data(self):
        ratings = load_data.get_rating(self.options)
        grouped = ratings.groupby(['value', 'rating_order']).count()
        grouped = grouped[['user_id']]
        grouped = grouped.reset_index()
        grouped = grouped.pivot(
            index='value',
            columns='rating_order',
            values='user_id')
        value_columns = grouped.columns
        grouped = grouped.fillna(0)
        grouped['All'] = 0
        for c in value_columns:
            grouped['All'] += grouped[c]
        for c in value_columns:
            grouped[c] = grouped[c] / grouped['All']
        grouped = grouped[value_columns]
        return grouped


class RatingByOrder(PlotCommand):
    stacked = True
    legend_loc = 'center right'

    def get_data(self):
        ratings = load_data.get_rating(self.options)
        grouped = ratings.groupby(['value', 'rating_order']).count()
        grouped = grouped[['user_id']]
        grouped = grouped.reset_index()
        grouped = grouped.pivot(
            index='rating_order',
            columns='value',
            values='user_id')
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


class RatingByOrderAb(PlotCommand):
    legend_loc = 'upper right'
    legend_alpha = True
    ylim = (0, 1)

    def get_data(self):
        all_ratings = load_data.get_rating_with_maps(self.options)
        data = []
        for i in AB_VALUES:
            ratings = all_ratings[all_ratings['experiment_setup_id'] == i]
            grouped = ratings.groupby(['value', 'rating_order']).count()
            grouped = grouped[['user_id']]
            grouped = grouped.reset_index()
            grouped = grouped.pivot(
                index='rating_order',
                columns='value',
                values='user_id')
            value_columns = grouped.columns
            grouped = grouped.fillna(0)
            grouped['All'] = 0
            for c in value_columns:
                grouped['All'] += grouped[c]
            for c in value_columns:
                grouped[c] = grouped[c] / grouped['All']
            grouped = grouped[value_columns]
            grouped.rename(columns=RATING_VALUES, inplace=True)
            data.append([grouped, AB_VALUES_SHORT[i]])
        return data


class RatingByAb(PlotCommand):
    stacked = True
    legend_loc = 'center right'
    figsize = (6, 4)

    def get_data(self):
        ratings = load_data.get_rating_with_maps(self.options)
        grouped = ratings.groupby(['experiment_setup_id', 'value']).count()
        grouped = grouped[['user_id']]
        grouped = grouped.reset_index()
        grouped = grouped.pivot(
            index='experiment_setup_id',
            columns='value',
            values='user_id')
        value_columns = grouped.columns
        grouped = grouped.fillna(0)
        grouped['All'] = 0
        for c in value_columns:
            grouped['All'] += grouped[c]
        for c in value_columns:
            grouped[c] = grouped[c] / grouped['All']
        grouped = grouped[value_columns]
        grouped.rename(columns=RATING_VALUES, inplace=True)
        grouped.rename(index=AB_VALUES_SHORT, inplace=True)
        grouped.sort_index(inplace=True)
        return grouped


class RatingByLastOrder(PlotCommand):
    stacked = True
    legend_loc = 'center right'

    def get_data(self):
        ratings = load_data.get_rating(self.options)
        grouped = ratings.groupby(['value', 'last_order']).count()
        grouped = grouped[['user_id']]
        grouped = grouped.reset_index()
        grouped = grouped.pivot(
            index='last_order',
            columns='value',
            values='user_id')
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


class RatingValueHistogram(PlotCommand):
    kind = 'hist'

    def get_data(self):
        ratings = load_data.get_rating(self.options)
        ratings = ratings[['value']]
        return ratings


class RatingOrderHistogram(PlotCommand):
    kind = 'hist'

    def get_data(self):
        ratings = load_data.get_rating(self.options)
        ratings = ratings[['rating_order']]
        return ratings


class RatingInTime(InTimeCommand):
    groupby_column = 'value'

    @property
    def answers(self):
        rating = load_data.get_rating(self.options)
        rating['time'] = rating['inserted']
        return rating


class RatingByDivider(DivisionCommand):
    stacked = True
    adjust_bottom = 0.2
    fontsize = 25

    def get_data(self):
        ratings = load_data.get_rating(self.options)
        ratings = ratings[['user_id', 'value']]
        div = divider.Divider.get_divider(self.options)
        answers = load_data.get_answers(self.options)
        new_column_name = div.column_name
        answers = div.divide(answers, new_column_name)
        users = answers.drop_duplicates(['user_id'])
        ratings = pd.merge(
            ratings,
            users,
            left_on=['user_id'],
            right_on=['user_id'],
        )
        ratings = ratings[['value', new_column_name, 'user_id']]
        grouped = ratings.groupby(['value', new_column_name]).count()
        grouped = grouped.reset_index()
        grouped = grouped.pivot(
            index=new_column_name,
            columns='value',
            values='user_id')
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


class RatingByDividerAb(DivisionCommand):
    adjust_bottom = 0.2
    legend_loc = 'upper right'
    legend_alpha = True
    subplots_adjust = dict(
        hspace=0.7,
        wspace=0.3,
        bottom=0.2,
    )
    figsize = (8, 4)
    ylim = (0, 1)

    def get_data(self):
        ratings = load_data.get_rating(self.options)
        ratings = ratings[['user_id', 'value']]
        div = divider.Divider.get_divider(self.options)
        answers = load_data.get_answers(self.options)
        new_column_name = div.column_name
        answers = div.divide(answers, new_column_name)
        users = answers.drop_duplicates(['user_id'])
        ratings = pd.merge(
            ratings,
            users,
            left_on=['user_id'],
            right_on=['user_id'],
        )
        data = []
        divider_values = sorted(ratings[new_column_name].unique().tolist())
        for i in divider_values:
            ratings_ab = ratings[ratings[new_column_name] == i]
            ratings_ab = ratings_ab[['value', 'experiment_setup_id', 'user_id']]
            grouped = ratings_ab.groupby(['value', 'experiment_setup_id']).count()
            grouped = grouped.reset_index()
            grouped = grouped.pivot(
                index='experiment_setup_id',
                columns='value',
                values='user_id')
            value_columns = grouped.columns
            grouped = grouped.fillna(0)
            grouped['All'] = 0
            for c in value_columns:
                grouped['All'] += grouped[c]
            for c in value_columns:
                grouped[c] = grouped[c] / grouped['All']
            grouped = grouped[value_columns]
            grouped.rename(columns=RATING_VALUES, inplace=True)
            grouped.rename(index=AB_VALUES_SHORT, inplace=True)
            grouped.index.name = 'Experiment condition'
            data.append([grouped,  "Is school user: " + str(i)])
        return data


class RatingByContextAb(DivisionCommand):
    adjust_bottom = 0.2
    legend_loc = 'upper right'
    legend_alpha = True
    ylim = (0, 1)
    subplots_adjust = dict(
        wspace=0.5,
        hspace=0.3,
    )
    figsize = (20, 15)

    def get_data(self):
        ratings = load_data.get_rating_with_maps(self.options)
        ratings['Context'] = ratings['context_name'] + ' - ' + ratings['term_type']
        print ratings
        top_contexts = ratings.groupby('Context').count()[['inserted']].sort(
            ['inserted'], ascending=[False]).head(12).reset_index()['Context'].tolist()
        data = []
        for context in top_contexts:
            ratings_ab = ratings[ratings['Context'] == context]
            ratings_ab = ratings_ab[['value', 'experiment_setup_id', 'user_id']]
            grouped = ratings_ab.groupby(['value', 'experiment_setup_id']).count()
            grouped = grouped.reset_index()
            grouped = grouped.pivot(
                index='experiment_setup_id',
                columns='value',
                values='user_id')
            value_columns = grouped.columns
            grouped = grouped.fillna(0)
            grouped['All'] = 0
            for c in value_columns:
                grouped['All'] += grouped[c]
            for c in value_columns:
                grouped[c] = grouped[c] / grouped['All']
            grouped = grouped[value_columns]
            grouped.rename(columns=RATING_VALUES, inplace=True)
            grouped.rename(index=AB_VALUES_SHORT, inplace=True)
            grouped.index.name = 'Experiment condition'
            data.append([grouped,  context])
        return data


class RatingBySuccess(DivisionCommand):
    adjust_bottom = 0.2
    legend_loc = 'upper right'
    legend_alpha = True
    subplots_adjust = dict(
        hspace=0.7,
        wspace=0.3,
        bottom=0.2,
    )
    ylim = (0, 1)

    def get_data(self):
        ratings = load_data.get_rating(self.options)
        ratings = ratings[['user_id', 'value']]
        answers = load_data.get_answers(self.options)
        grouped = answers.groupby(['user_id']).mean()
        grouped = grouped[['correct']]
        grouped = grouped.reset_index()
        ratings = pd.merge(
            ratings,
            grouped,
            left_on=['user_id'],
            right_on=['user_id'],
        )
        ratings['Success rate'] = ratings['correct'].map(lambda x: round(x, 1))
        ratings = ratings[['value', 'Success rate', 'user_id']]
        grouped = ratings.groupby(['value', 'Success rate']).count()
        grouped = grouped.reset_index()
        grouped = grouped.pivot(
            index='Success rate',
            columns='value',
            values='user_id')
        value_columns = grouped.columns
        grouped = grouped.fillna(0)
        grouped['All'] = 0
        for c in value_columns:
            grouped['All'] += grouped[c]
        for c in value_columns:
            grouped[c] = grouped[c] / grouped['All']
        grouped = grouped[value_columns]
        grouped.rename(columns=RATING_VALUES, inplace=True)
        grouped.rename(index=AB_VALUES_SHORT, inplace=True)
        return grouped


class RatingBySuccessAb(PlotCommand):
    adjust_bottom = 0.2
    legend_loc = 'upper right'
    legend_alpha = True
    subplots_adjust = dict(
        hspace=0.5,
        wspace=0.3,
        bottom=0.2,
    )
    ylim = (0, 1)

    def get_data(self):
        ratings = load_data.get_rating(self.options)
        ratings = ratings[['user_id', 'value']]
        answers = load_data.get_answers(self.options)
        grouped = answers.groupby(['user_id']).mean()
        grouped = grouped[['correct', 'experiment_setup_id']]
        grouped = grouped.reset_index()
        ratings = pd.merge(
            ratings,
            grouped,
            left_on=['user_id'],
            right_on=['user_id'],
        )
        ratings = ratings[ratings['correct'] >= 0.4]
        ratings['Global success rate'] = ratings['correct'].map(
            lambda x: round(x * 1, 1) / 1.0)
        ratings = ratings[['value', 'Global success rate', 'user_id', 'experiment_setup_id']]

        data = []
        ab_values = sorted(ratings['experiment_setup_id'].unique().tolist())
        for i in ab_values:
            ratings_ab = ratings[ratings['experiment_setup_id'] == i]
            grouped = ratings_ab.groupby(['value', 'Global success rate']).count()
            grouped = grouped.reset_index()
            grouped = grouped.pivot(
                index='Global success rate',
                columns='value',
                values='user_id')
            value_columns = grouped.columns
            grouped = grouped.fillna(0)
            grouped['All'] = 0
            for c in value_columns:
                grouped['All'] += grouped[c]
            for c in value_columns:
                grouped[c] = grouped[c] / grouped['All']
            grouped = grouped[value_columns]
            grouped.rename(columns=RATING_VALUES, inplace=True)
            grouped.rename(index=AB_VALUES_SHORT, inplace=True)
            data.append([grouped, "Experimental condition: " + AB_VALUES[i]])
        return data


class RatingByRollingSuccessAb(PlotCommand):
    adjust_bottom = 0.2
    legend_loc = 'upper right'
    legend_alpha = True
    subplots_adjust = dict(
        hspace=0.5,
        wspace=0.3,
        bottom=0.2,
    )
    ylim = (0, 1)

    def get_data(self):
        ratings = load_data.get_rating_with_rolling_success(self.options)
        ratings = ratings[ratings['rolling_success'] >= 0.4]
        ratings['Last 10 answers success rate'] = ratings['rolling_success'].map(
            lambda x: round(x * 2, 1) / 2.0)
        ratings = ratings[['value', 'Last 10 answers success rate', 'user_id', 'experiment_setup_id']]

        data = []
        ab_values = sorted(ratings['experiment_setup_id'].unique().tolist())
        for i in ab_values:
            ratings_ab = ratings[ratings['experiment_setup_id'] == i]
            grouped = ratings_ab.groupby(['value', 'Last 10 answers success rate']).count()
            grouped = grouped.reset_index()
            grouped = grouped.pivot(
                index='Last 10 answers success rate',
                columns='value',
                values='user_id')
            value_columns = grouped.columns
            grouped = grouped.fillna(0)
            grouped['All'] = 0
            for c in value_columns:
                grouped['All'] += grouped[c]
            for c in value_columns:
                grouped[c] = grouped[c] / grouped['All']
            grouped = grouped[value_columns]
            grouped.rename(columns=RATING_VALUES, inplace=True)
            grouped.rename(index=AB_VALUES_SHORT, inplace=True)
            data.append([grouped, "Experimental condition: " + AB_VALUES[i]])
        return data


class RatingByRollingSuccessContexts(PlotCommand):
    adjust_bottom = 0.2
    legend_loc = 'upper right'
    legend_alpha = True
    ylim = (0, 1)
    subplots_adjust = dict(
        wspace=0.5,
        hspace=0.3,
    )
    figsize = (20, 15)

    def get_data(self):
        ratings = load_data.get_rating_with_rolling_success(self.options)
        ratings = ratings[ratings['rolling_success'] >= 0.4]
        ratings['Success rate'] = ratings['rolling_success'].map(
            lambda x: round(x * 2, 1) / 2.0)
        ratings['Context'] = ratings['context_name'] + ' - ' + ratings['term_type']
        top_contexts = ratings.groupby('Context').count()[['inserted']].sort(
            ['inserted'], ascending=[False]).head(12).reset_index()['Context'].tolist()
        ratings = ratings[['value', 'Success rate', 'user_id', 'Context']]

        data = []
        for context in top_contexts:
            ratings_ab = ratings[ratings['Context'] == context]
            grouped = ratings_ab.groupby(['value', 'Success rate']).count()
            grouped = grouped.reset_index()
            grouped = grouped.pivot(
                index='Success rate',
                columns='value',
                values='user_id')
            value_columns = grouped.columns
            grouped = grouped.fillna(0)
            grouped['All'] = 0
            for c in value_columns:
                grouped['All'] += grouped[c]
            for c in value_columns:
                grouped[c] = grouped[c] / grouped['All']
            grouped = grouped[value_columns]
            grouped.rename(columns=RATING_VALUES, inplace=True)
            grouped.rename(index=AB_VALUES_SHORT, inplace=True)
            data.append([grouped, context])
        return data


class RatingByDividers(PlotCommand):
    stacked = True
    adjust_bottom = 0.2
    legend_loc = 'center right'

    def get_data(self):
        self.options.divider = 'all'
        divs = divider.Divider.get_divider(self.options)
        ret = None
        for i in divs:
            div = divs[i](self.options)
            print div.column_name
            data = self._get_data(div)
            ret = data if ret is None else ret.append(data)
        return ret

    def _get_data(self, div):
        ratings = load_data.get_rating(self.options)
        ratings = ratings[['user_id', 'value']]
        answers = load_data.get_answers(self.options)
        new_column_name = div.column_name
        answers = div.divide(answers, new_column_name)
        users = answers.drop_duplicates(['user_id'])
        ratings = pd.merge(
            ratings,
            users,
            left_on=['user_id'],
            right_on=['user_id'],
        )
        ratings = ratings[['value', new_column_name, 'user_id']]
        grouped = ratings.groupby(['value', new_column_name]).count()
        grouped = grouped.reset_index()
        grouped = grouped.pivot(
            index=new_column_name,
            columns='value',
            values='user_id')
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


class SuccessByDividers(PlotCommand):
    stacked = True
    adjust_bottom = 0.2

    def get_data(self):
        self.options.divider = 'all'
        divs = divider.Divider.get_divider(self.options)
        ret = None
        for i in divs:
            div = divs[i](self.options)
            print div.column_name
            data = self._get_data(div)
            ret = data if ret is None else ret.append(data)
        return ret

    def _get_data(self, div):
        answers = load_data.get_answers(self.options)
        new_column_name = div.column_name
        answers = div.divide(answers, new_column_name)
        answers = answers[['correct', new_column_name, 'id']]
        grouped = answers.groupby(['correct', new_column_name]).count()
        grouped = grouped.reset_index()
        grouped = grouped.pivot(
            index=new_column_name,
            columns='correct',
            values='id')
        value_columns = grouped.columns
        grouped = grouped.fillna(0)
        grouped['All'] = 0
        for c in value_columns:
            grouped['All'] += grouped[c]
        for c in value_columns:
            grouped[c] = grouped[c] / grouped['All']
        grouped = grouped[value_columns]
        grouped.rename(
            columns={True: 'Correct answers', False: 'Incorrect answers'},
            inplace=True)
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


class FeedbackByType(PlotCommand):
    kind = 'pie'
    subplots = True
    legend_bbox = (0.75, -01.0)

    def get_data(self):
        feedback = load_data.get_feedback_data_with_type()
        print len(feedback)
        grouped = feedback.groupby(['type']).count()
        grouped = grouped[['time']]
        grouped = grouped.sort('time', ascending=False)
        """
        grouped = grouped.reset_index()
        grouped['type'] = grouped['type'] + grouped['time'].apply(
            lambda x: ' (' + str(x) + ')')
        grouped = grouped.set_index('type')
        """
        grouped.columns = ['']
        return grouped


class FeedbackWordCountHistogram(PlotCommand):
    kind = 'hist'

    def get_data(self):
        feedback = load_data.get_feedback_data()
        feedback['log_10(word_count)'] = feedback['word_count'].map(lambda x: math.log(x, 10))
        feedback = feedback[['log_10(word_count)']]
        return feedback


class AnswerCountByUserAb(PlotCommand):

    def get_data(self):
        answers = load_data.get_answers(self.options)
        grouped = answers.groupby(['user_id', 'experiment_setup_id']).count()
        grouped = grouped[['id']]
        grouped.columns = ['answer_count']
        grouped = grouped.reset_index()
        grouped = grouped.groupby(['experiment_setup_id']).mean()
        grouped = grouped[['answer_count']]
        grouped = grouped.reset_index()
        grouped = grouped.set_index('experiment_setup_id')
        grouped.rename(index=AB_VALUES_SHORT, inplace=True)
        grouped.sort_index(inplace=True)
        return grouped


class AnswerCountBySessionAb(PlotCommand):

    def get_data(self):
        answers = load_data.get_answers(self.options)
        grouped = answers.groupby(['session_id', 'experiment_setup_id']).count()
        grouped = grouped[['id']]
        grouped.columns = ['answer_count']
        grouped = grouped.reset_index()
        grouped = grouped.groupby(['experiment_setup_id']).mean()
        grouped = grouped[['answer_count']]
        grouped = grouped.reset_index()
        grouped = grouped.set_index('experiment_setup_id')
        grouped.rename(index=AB_VALUES_SHORT, inplace=True)
        grouped.sort_index(inplace=True)
        return grouped


class AnswersToRatingsRatio(PlotCommand):
    adjust_bottom = 0.2

    def get_data(self):
        answers = load_data.get_answers(self.options)
        grouped = answers.groupby(['user_id', 'experiment_setup_id']).count()
        grouped = grouped[['id']]
        grouped.columns = ['answer_count']
        grouped['answer_count'] = grouped['answer_count'].map(
            lambda x: bisect.bisect_left([30, 70, 120, 200], x))
        grouped = grouped.reset_index()
        print grouped
        ratings = load_data.get_rating(self.options)
        ratings = ratings.groupby(['user_id']).count()
        ratings = ratings[['value']]
        ratings.columns = ['rating_count']
        ratings = ratings.reset_index()
        grouped = pd.merge(
            ratings,
            grouped,
            left_on=['user_id'],
            right_on=['user_id'],
        )
        grouped['diff'] = grouped['answer_count'] - grouped['rating_count']
        grouped = grouped.groupby(['experiment_setup_id', 'diff']).count()
        grouped = grouped[['user_id']]
        grouped.columns = ['answer_count - rating_count']
        grouped.rename(index=AB_VALUES_SHORT, inplace=True)
        grouped.sort_index(inplace=True)
        return grouped
