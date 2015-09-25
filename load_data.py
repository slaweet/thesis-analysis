# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
from main import DATA_DIR
import datetime
import bisect

RATING_INTERVALS = [30, 70, 120, 200]


def get_mnemonics(options):
    data_file = 'mnemonics_cs.csv'
    data_file = os.path.join(options.data_dir, data_file)
    mnemonics = pd.read_csv(data_file, dtype=unicode, sep='\t', header=None)
    mnemonics = mnemonics[mnemonics[1].notnull()]
    places = read_csv(DATA_DIR + 'geography.place.csv')
    data = pd.merge(
        places,
        mnemonics,
        left_on=['code'],
        right_on=[0],
    )
    data = data[['id']]
    return data


def get_rating(options):
    ratings = read_csv(options.ratings)
    ratings = ratings.sort(['user_id', 'inserted'], ascending=True)
    rating_orders = []
    order_by_user = {}
    grouped = ratings.groupby('user_id')
    for i, g in grouped:
        order_by_user[i] = 0
        for j, row in g['value'].iteritems():
            order_by_user[i] += 1
            rating_orders.append(order_by_user[i])
    ratings['rating_order'] = pd.Series(rating_orders, index=ratings.index)
    ratings['last_order'] = ratings['user_id'].map(lambda x: order_by_user[x])
    ratings = ratings[ratings['last_order'] <= 4]
    return ratings


def get_answers_with_flashcards_and_orders(options):
    answers_with_flashcards = get_answers_with_flashcards(options)
    answers_with_flashcards = answers_with_flashcards.sort(
        ['user_id', 'time'], ascending=True)

    answer_orders = []
    order_by_user = {}
    answers_with_flashcards = answers_with_flashcards[answers_with_flashcards['context_id'] != 17]
    answers_with_flashcards = answers_with_flashcards[answers_with_flashcards['context_id'].isin([1, 86])]
    grouped = answers_with_flashcards.groupby('user_id')
    for i, g in grouped:
        order_by_user[i] = 0
        for j, row in g['item_id'].iteritems():
            order_by_user[i] += 1
            answer_orders.append(order_by_user[i])
    answers_with_flashcards['answer_order'] = pd.Series(
        answer_orders, index=answers_with_flashcards.index)
    return answers_with_flashcards


def get_rating_with_maps(options):
    ratings = get_rating(options)
    ratings['answer_order'] = ratings['rating_order'].map(lambda x: RATING_INTERVALS[x - 1])
    answers = get_answers_with_flashcards_and_orders(options)

    ratings_with_maps = pd.merge(
        ratings,
        answers,
        left_on=['user_id', 'answer_order'],
        right_on=['user_id', 'answer_order'],
    )
    return ratings_with_maps


def get_answers(options, strip_times=False, strip_less_than_10=False):
    col_types = {
        'user_id': np.uint32,
        'id': np.uint32,
        'item_id': np.uint32,
        'item_asked_id': np.uint32,
        'item_answered_id': np.float32,  # because of NAs
        'type': np.uint8,
        'response_time': np.uint32,
        'number_of_options': np.uint8,
        'place_map': np.float16,       # because of NAs
        'ip_address': str,
        'language': str,
        'test_id': np.float16          # because of NAs
    }
    answers = read_csv(options.answers, col_types)
    answers['correct'] = answers['item_id'] == answers['item_answered_id']
    if strip_times:
        answers = answers[answers['response_time'] > 0]
        answers = answers[answers['response_time'] < 30000]
    if strip_less_than_10:
        grouped = answers.groupby('user').count()
        grouped = grouped.reset_index()
        more10 = grouped[grouped['inserted'] >= 10]['user']
        answers = answers[answers['user'].isin(more10)]
    return answers


def get_ab_values(options):
    col_types = np.int64
    ab_values = read_csv(options.ab_values, options, col_types)
    return ab_values


def get_answers_with_ab(options, answers=None):
    if answers is None:
        answers = get_answers(options)
    ab_values = get_ab_values(options)
    answers_with_ab = pd.merge(
        answers,
        ab_values,
        left_on=['id'],
        right_on=['answer'],
    )
    return answers_with_ab


def get_answers_with_map_grouped(options):
    answers_with_maps = get_answers_with_flashcards(options)
    answers_with_maps = answers_with_maps.groupby(['map_name'])
    return answers_with_maps


def get_answers_with_flashcards(options):
    answers = get_answers(options)
    flashcards = get_flashcards(options)
    answers_with_flashcards = pd.merge(
        answers,
        flashcards,
        left_on=['item_id'],
        right_on=['item_id'],
    )
    if options.context_name is not None:
        answers_with_flashcards = answers_with_flashcards[
            answers_with_flashcards['context_name'] == options.context_name]
    if options.term_type is not None:
        answers_with_flashcards = answers_with_flashcards[
            answers_with_flashcards['term_type'] == options.term_type]
    return answers_with_flashcards


def get_flashcards(options):
    flashcards = read_csv(DATA_DIR + 'flashcards.csv')
    contexts = flashcards.groupby(['context_name', 'term_type']).count()
    contexts = contexts[['item_id']]
    contexts = contexts.reset_index()
    contexts.rename(columns={'item_id': 'context_item_count'}, inplace=True)
    step = 20
    contexts['context_size'] = contexts['context_item_count'].map(
        lambda x: bisect.bisect_left(
            range(step, 160, step), x) * step + step / 2)
    flashcards = pd.merge(
        contexts,
        flashcards,
        left_on=['context_name', 'term_type'],
        right_on=['context_name', 'term_type'],
    )
    return flashcards


def get_prior_skills(options):
    col_types = {
        'user': np.uint32,
        'value': np.float16,
    }
    ab_values = read_csv(DATA_DIR + 'geography.skill.csv', col_types)
    return ab_values


FEEDBACK_TYPES = {
    "0": "Other",  # "spam",
    "1": "Praise",
    "2": "Other",  # "wrong missing",
    "3": "Content request",
    "4": "Functionality request",
    "5": "Error in content",
    "6": "Error in \nfunctionality",
    "7": "No information value",
    "8": "Other",
}


def get_feedback_data_with_type():
    data = read_csv('data/feedback_with_types_added.csv', unicode)
    data['type'] = data['type'].apply(lambda x: FEEDBACK_TYPES[x])
    return data


def get_feedback_data():
    data = read_csv('data/messages.csv', unicode)
    data['text'] = data['message'].apply(parse_msg)
    data['inserted'] = data['date'].apply(parse_date)
    data = data[['inserted', 'text']]
    data['text_length'] = data['text'].apply(lambda x: len(x))
    data['word_count'] = data['text'].apply(lambda x: len(x.split()))
    data['const'] = 'Number of received messages'
    data['id'] = data['text_length']
    data = data.drop_duplicates('text')
    data = data[data['word_count'] > 1]
    data = data[data['text_length'] < 10000]
    data.to_csv(DATA_DIR + 'feedback_with_types.csv')
    data = data.sort('word_count', ascending=False)
    return data


def count_keywords(data):
    data = data.sort('word_count')
    keywords = ['super', 'chyb', 'řek', 'pohoří', 'moř']
    counts = {}
    for keyword in keywords:
        data[keyword] = data['text'].apply(lambda x: keyword in x.lower())
        df = data[data[keyword]]
        counts[keyword] = len(df)
    return counts


def parse_msg(text):
    parts = text.split('##')
    if len(parts) == 5:
        return parts[2].strip()

    parts = text.split('feedback:\r\n')
    if len(parts) == 2:
        parts = parts[1].split('\r\nemail:')
        if len(parts) == 2:
            return parts[0].strip()
    return ''


def parse_date(x):
    date, time = x.split(' ')
    datetime_list = map(int, date.split('.')[::-1] + time.split(':'))
    return datetime.datetime(*datetime_list).isoformat(' ')


def read_csv(data_file, dtype=None):
    data = pd.read_csv(data_file, dtype=dtype)
    return data
