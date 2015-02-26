# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import os

DATA_DIR = 'data/'
PLOT_DIR = 'plot/'


def places_by_map(options, command_name):
    maps = get_maps(options)
    maps = maps.groupby(['map_name']).count()
    maps = maps[['place']]
    maps = maps.sort('place', ascending=False)
    maps.columns = ['Number Of Places']
    generate_graph(maps, command_name)


def places_by_map_and_type(options, command_name):
    maps = get_maps(options)
    maps = maps.groupby(['map_name', 'place_type']).count()
    maps = maps[['place']]
    maps = maps.sort('place', ascending=False)
    maps.columns = ['Number Of Places']
    generate_graph(maps, command_name)


def success_by_map(options, command_name):
    answers_with_maps = get_answers_with_map(options)
    success_rate_by_map = answers_with_maps.mean().sort('correct')
    generate_graph(success_rate_by_map, command_name)


def get_answers(options):
    answers = read_csv(options.answers, options)
    answers['correct'] = answers['place_asked'] == answers['place_answered']
    return answers


def get_answers_with_map(options):
    answers = get_answers(options)
    maps = get_maps(options)
    answers_with_maps = pd.merge(
        answers,
        maps,
        left_on=['place_asked'],
        right_on=['place'],
    )
    answers_with_maps = answers_with_maps.groupby(['map_name'])
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
    maps = maps[maps.type == '1']
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


def get_answers_by_map(data):
    pass


def generate_graph(data, name):
    print len(data)
    print data.head()
    data.plot(kind='bar')
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.3)
    plt.savefig(PLOT_DIR + name + '.png')
    plt.show()


def read_csv(data_file, options):
    data_file = os.path.join(options.data_dir, data_file)
    data = pd.read_csv(data_file, dtype=unicode)
    if options.verbose:
        print "File", data_file, "data lenght", len(data)
        print data.head()
    return data
