#!/usr/bin/python
# -*- coding: utf-8 -*-

import optparse
import commands
import re

DATA_DIR = 'data/'


def main():
    p = optparse.OptionParser()

    p.add_option("-a", "--answers", action="store",
                 dest="answers", default=DATA_DIR + 'geography.answer.csv',
                 help="path to file with answers inside data dir")

    p.add_option("-d", "--data-dir", action="store",
                 dest="data_dir", default=DATA_DIR,
                 help="path to directory with data")

    p.add_option("-i", "--divider", action="store",
                 dest="divider", default='school_divider',
                 help="lower_case_hyphenated name of class to separate the answers")

    p.add_option("-r", "--ratings", action="store",
                 dest="ratings", default=DATA_DIR + 'feedback.rating.csv',
                 help="path to file with answers inside data dir")

    p.add_option("-v", "--verbose", action="store_true",
                 dest="verbose", default=False,
                 help="print extra stuff")
    options, arguments = p.parse_args()

    possible_commands = dict([
        (commands.Command.name(c), c) for c in all_subclasses(commands.Command)])
    if len(arguments) == 0:
        print_error('No command specified')
        print_help(possible_commands)
        return

    command = arguments[0]
    if command in possible_commands:
        possible_commands[command](options).execute()
    else:
        print_error('Unknown command: ' + command)
        print_help(possible_commands)


def print_help(possible_commands):
    print 'Available commands:'
    print '  ' + '\n  '.join(possible_commands.keys())


def all_subclasses(cls):
    return cls.__subclasses__() + [g for s in cls.__subclasses__()
                                   for g in all_subclasses(s)]


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


def print_ok(msg):
    print bcolors.OKGREEN + msg + bcolors.ENDC


def print_error(msg):
    print bcolors.FAIL + msg + bcolors.ENDC


def convert_from_cammel_case(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

if __name__ == '__main__':
    main()
