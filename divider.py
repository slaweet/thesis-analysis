class Divider(object):
    min_treshold = 1
    max_treshold = 20

    def divide(self, answers, new_column_name, treshold):
        pass


class SchoolDivider(Divider):
    def divide(self, answers, new_column_name, treshold=5):
        classroom_size = treshold
        answers = answers[~answers['ip_address'].isin([None, ''])]
        classroom_users = [
            user
            for ip, users in (
                answers.sort('id').drop_duplicates('user').
                groupby('ip_address').
                apply(lambda x: x['user'].unique()).
                to_dict().
                items())
            for user in users
            if len(users) >= classroom_size
        ]
        answers[new_column_name] = answers['user'].isin(classroom_users)
        return answers


class NewVsReturning(Divider):
    max_treshold = 20

    def divide(self, answers, new_column_name, treshold=5):
        returning = []
        answers = answers.sort(['inserted'], ascending=True)
        users = answers.groupby('user')
        for name, group in users:
            if group.values[0][6][:treshold] != group.values[-1][6][:treshold]:
                returning.append(name)

        answers[new_column_name] = answers['user'].isin(returning)
        return answers


class MorningVsEvening(Divider):
    max_treshold = 25

    def divide(self, answers, new_column_name, treshold=5):
        answers = answers[~answers['inserted'].isin([None, ''])]
        answers['time'] = answers['inserted'].map(to_seconds)
        users = answers.sort(['id'], ascending=True).groupby('user').mean()
        users = users.reset_index().drop_duplicates('user')
        users = users.set_index(['user'])
        times_by_user = users['time']
        times_by_user = times_by_user.to_dict().items()
        after_users = [
            user
            for (user, seconds)
            in times_by_user
            if seconds > treshold * 3600]
        answers[new_column_name] = answers['user'].isin(after_users)
        return answers


def to_seconds(datetime_string):
    (h, m, s) = datetime_string[11:].split(':')
    result = int(h) * 3600 + int(m) * 60 + int(s)
    return result
