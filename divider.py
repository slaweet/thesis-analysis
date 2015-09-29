import utils
import load_data
import math


class Divider(object):
    min_treshold = 1
    max_treshold = 20
    step = 1
    column_name = 'Not set'

    def __init__(self, options=None):
        self.options = options

    @staticmethod
    def get_divider(options):
        possible_dividers = dict([
            (utils.convert_from_cammel_case(c.__name__), c)
            for c in utils.all_subclasses(Divider)])
        if options.divider == 'all':
            return possible_dividers
        if options.divider not in possible_dividers:
            raise Exception('Invalid divider name: ' + options.divider)
        div = possible_dividers[options.divider](options)
        return div

    def divide(self, answers, new_column_name, treshold=None):
        if treshold is None:
            positive_users = self.get_positive_users(answers)
        else:
            positive_users = self.get_positive_users(answers, treshold)
        answers[new_column_name] = answers['user_id'].isin(positive_users)
        return answers

    def get_positive_users(self, answers, treshold):
        raise NotImplementedError('divide must be implemented in subclass')


class SchoolDivider(Divider):
    column_name = 'Is School User'
    max_treshold = 40
    step = 2

    def get_positive_users(self, answers, treshold=25):
        classroom_size = treshold
        answers = answers[~answers['ip_address'].isin([None, ''])]
        classroom_users = [
            user
            for ip, users in (
                answers.sort('id').drop_duplicates('user_id').
                groupby('ip_address').
                apply(lambda x: x['user_id'].unique()).
                to_dict().
                items())
            for user in users
            if len(users) >= classroom_size
        ]
        return classroom_users


"""
class SchoolDividerImproved(Divider):
    column_name = 'Is School User (timing)'

    def divide(self, answers, new_column_name, treshold=5):
        answers = answers[~answers['ip_address'].isin([None, ''])]
        users = answers.sort('id').drop_duplicates('user_id')[[
            'ip_address', 'time', 'user_id']]
        users['date'] = users['time'].map(lambda x: x[:10])
        grouped = users.groupby(['ip_address', 'date']).apply(
            lambda x: (x['user_id'].unique(), x['time'].unique())).to_dict()
        classroom_users = []
        for ip, data in grouped.iteritems():
            if len(data[0]) >= treshold:
                if len(data[0]) < 7:
                    print list(data[0]), list(data[1])
                classroom_users = classroom_users + list(data[0])
        # print len(users)
        answers[new_column_name] = answers['user_id'].isin(classroom_users)
        return answers
"""


class NewVsReturning(Divider):
    max_treshold = 20
    column_name = 'Is Returning User'

    def get_positive_users(self, answers, treshold=10):
        returning = []
        # answers.sort(['time'], ascending=True, inplace=True)
        users = answers[['user_id', 'time']].groupby('user_id')
        for name, group in users:
            # print group.values[0][1][:treshold], group.values[-1][1][:treshold]
            if group.values[0][1][:treshold] != group.values[-1][1][:treshold]:
                returning.append(name)
        return returning


class MorningVsEvening(Divider):
    max_treshold = 25
    column_name = 'Average time > 16:00'

    def get_positive_users(self, answers, treshold=16):
        answers = answers[~answers['time'].isin([None, ''])]
        answers['time'] = answers['time'].map(to_seconds)
        users = answers.sort(['id'], ascending=True).groupby('user_id').mean()
        users = users.reset_index().drop_duplicates('user_id')
        users = users.set_index(['user_id'])
        times_by_user = users['time']
        times_by_user = times_by_user.to_dict().items()
        after_users = [
            user
            for (user, seconds)
            in times_by_user
            if seconds > treshold * 3600]
        answers = answers.drop('time', 1)
        return after_users


"""
class HasTargetAdjusted(Divider):
    max_treshold = 1
    column_name = 'Has Adjusted Target Probability'

    def get_positive_users(self, answers, treshold=None):
        answers_with_ab = load_data.get_answers_with_ab(self.options, answers)
        answers_with_ab = answers_with_ab[answers_with_ab['value'] == 16]
        answers_with_ab = answers_with_ab.drop_duplicates('user_id')
        return answers_with_ab['user_id']


class RecommendsByRandom(Divider):
    max_treshold = 1
    column_name = 'Uses random question selection'

    def get_positive_users(self, answers, treshold=None):
        answers_with_ab = load_data.get_answers_with_ab(self.options, answers)
        answers_with_ab = answers_with_ab[answers_with_ab['value'] == 2]
        answers_with_ab = answers_with_ab.drop_duplicates('user_id')
        return answers_with_ab['user_id']


class HasHighPriorSkill(Divider):
    min_treshold = -5
    max_treshold = 6
    column_name = 'Has High prior skill'

    def get_positive_users(self, answers, treshold=0):
        skills = load_data.get_prior_skills(self.options)
        skills = skills[skills['value'] > treshold]
        return skills['user']
"""


class IsFast(Divider):
    max_treshold = 10
    column_name = 'Responds quickly'

    def get_positive_users(self, answers, treshold=3.5):
        users = answers.groupby(['user_id']).median()
        users = users.reset_index().drop_duplicates('user_id')
        users = users.set_index(['user_id'])
        times_by_user = users['response_time']
        times_by_user = times_by_user.to_dict().items()
        fast_users = [
            user
            for (user, seconds)
            in times_by_user
            if seconds < treshold * 1000]
        return fast_users


"""
class HighResponseTimeVariance(Divider):
    min_treshold = 0
    max_treshold = 10
    column_name = 'High Response Variance'

    def get_positive_users(self, answers, treshold=1):
        answers = answers[answers['response_time'] > 0]
        answers['response_time'] = answers['response_time'].apply(
            lambda x: math.log(x, 2))
        users = answers.groupby(['user_id']).var()
        users = users.reset_index().drop_duplicates('user_id')
        users = users.set_index(['user_id'])
        times_by_user = users['response_time']
        times_by_user = times_by_user.to_dict().items()
        fast_users = [
            user
            for (user, seconds)
            in times_by_user
            if seconds < treshold]
        return fast_users
"""


def to_seconds(datetime_string):
    (h, m, s) = datetime_string[11:19].split(':')
    result = int(h) * 3600 + int(m) * 60 + int(s)
    return result
