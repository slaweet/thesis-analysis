#!/usr/bin/python
# -*- coding: utf-8 -*-

import optparse
import commands
import utils

DATA_DIR = 'data/'


def main():
    p = optparse.OptionParser()

    p.add_option("-a", "--answers", action="store",
                 dest="answers", default=DATA_DIR + 'answers.csv',
                 help="path to file with answers inside data dir")

    p.add_option("-b", "--ab_values", action="store",
                 dest="ab_values", default=DATA_DIR + 'geography.answer_ab_values.csv',
                 help="path to file with values of a/b experiments")

    p.add_option("-d", "--data-dir", action="store",
                 dest="data_dir", default=DATA_DIR,
                 help="path to directory with data")

    p.add_option("-i", "--divider", action="store",
                 dest="divider", default='school_divider',
                 help="lower_case_hyphenated name of class to separate the answers")

    p.add_option("-r", "--ratings", action="store",
                 dest="ratings", default=DATA_DIR + 'ratings.csv',
                 help="path to file with answers inside data dir")

    p.add_option("-v", "--verbose", action="store_true",
                 dest="verbose", default=False,
                 help="print extra stuff")

    p.add_option("-p", "--production", action="store_true",
                 dest="production", default=False,
                 help="create production quality graphs")

    p.add_option("-c", "--context_name", action="store",
                 dest="context_name", default=None,
                 help="use only answers on given context_name")

    p.add_option("-t", "--term_type", action="store",
                 dest="term_type", default=None,
                 help="use only answers on given type of terms")

    p.add_option("-u", "--use_cached_data", action="store_true",
                 dest="use_cached_data", default=None,
                 help="use data from last run of this command")

    p.add_option("-n", "--no_cache", action="store_true",
                 dest="no_cache", default=None,
                 help="don't use data cached by functions in load_data.py")

    p.add_option("-e", "--hide_plots", action="store_true",
                 dest="hide_plots", default=False,
                 help="don't show generated plot, just save it")

    options, arguments = p.parse_args()

    possible_commands = dict([
        (commands.Command.name(c), c) for c in utils.all_subclasses(commands.Command)])
    if len(arguments) == 0:
        print_error('No command specified')
        print_help(possible_commands)
        return

    command = arguments[0]
    if command == 'all':
        errors = []
        for i, Cmd in possible_commands.iteritems():
            print 'processing', i
            try:
                cmd = Cmd(options, show_plots=False)
                if cmd.active:
                    cmd.execute()
                else:
                    print_error('Command inactive: ' + i)
            except Exception, e:
                errors.append(i + ': ' + str(e))
                print_error('Failed command: ' + i + ': ' + str(e))
        if len(errors) != 0:
            print_error('Failed commands: \n' + '\n\n'.join(errors))
    elif command in possible_commands:
        possible_commands[command](options).execute()
    else:
        print_error('Unknown command: ' + command)
        print_help(possible_commands)


def print_help(possible_commands):
    print 'Available commands:'
    print '  ' + '\n  '.join(sorted(possible_commands.keys()))


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


if __name__ == '__main__':
    main()
