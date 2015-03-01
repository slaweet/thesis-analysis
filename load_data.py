# -*- coding: utf-8 -*-

import pandas as pd
import os


def get_mnemonics(options):
    data_file = 'mnemonics_cs.csv'
    data_file = os.path.join(options.data_dir, data_file)
    mnemonics = pd.read_csv(data_file, dtype=unicode, sep='\t', header=None)
    mnemonics = mnemonics[mnemonics[1].notnull()]
    places = read_csv('geography.place.csv', options)
    data = pd.merge(
        places,
        mnemonics,
        left_on=['code'],
        right_on=[0],
    )
    data = data[['id']]
    return data


def get_rating(options):
    ratings = read_csv(options.ratings, options)
    answers_with_maps = get_answers_with_map(options)
    ratings_with_maps = pd.merge(
        ratings,
        answers_with_maps,
        left_on=['user'],
        right_on=['user'],
    )
    return ratings_with_maps


def get_answers(options):
    answers = read_csv(options.answers, options)
    answers['correct'] = answers['place_asked'] == answers['place_answered']
    return answers


def get_answers_with_map_grouped(options):
    answers_with_maps = get_answers_with_map(options)
    answers_with_maps = answers_with_maps.groupby(['map_name'])
    return answers_with_maps


def get_answers_with_map(options):
    answers = get_answers(options)
    maps = get_maps(options)
    answers_with_maps = pd.merge(
        answers,
        maps,
        left_on=['place_asked'],
        right_on=['place'],
    )
    return answers_with_maps


def get_maps(options):
    places = read_csv('geography.place.csv', options)
    placerelations = read_csv('geography.placerelation.csv', options)
    placerelations_related = read_csv(
        'geography.placerelation_related_places.csv', options)
    maps = pd.merge(
        placerelations,
        placerelations_related,
        left_on=['id'],
        right_on=['placerelation'],
    )
    maps = maps[maps.type == 1]
    maps = pd.merge(
        maps,
        places,
        left_on=['place_x'],
        right_on=['id'],
    )
    maps = pd.merge(
        maps,
        places,
        left_on=['place_y'],
        right_on=['id'],
    )
    # print maps.head()
    maps = maps[['place_x', 'place_y', 'name_en_y', 'name_en_x', 'type']]
    maps.columns = ['map', 'place', 'place_name', 'map_name', 'place_type']
    return maps

import numpy


def read_csv(data_file, options):
    data_file = os.path.join(options.data_dir, data_file)
    data = pd.read_csv(data_file)  # , dtype={'user': numpy.float64})
    if options.verbose:
        print "File", data_file, "data lenght", len(data)
        print data.head()
    return data
