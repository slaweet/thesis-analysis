#!/usr/bin/python
# -*- coding: utf-8 -*-

import optparse
import commands


def main():
    p = optparse.OptionParser()
    p.add_option("-a", "--answers", action="store",
                 dest="answers", default='geography.answer.csv',
                 help="path to file with answers inside data dir")
    p.add_option("-d", "--data-dir", action="store",
                 dest="data_dir", default='data',
                 help="path to directory with data")
    p.add_option("-v", "--verbose", action="store_true",
                 dest="verbose", default=False,
                 help="print extra stuff")
    options, arguments = p.parse_args()
    if len(arguments) == 0:
        print 'too few args'
        return
    command = arguments[0]
    if command == 'success_by_map':
        commands.success_by_map(options, command)
    elif command == 'places_by_map':
        commands.places_by_map(options, command)
    elif command == 'places_by_map_and_type':
        commands.places_by_map_and_type(options, command)
    else:
        print 'unknown argument: ', command

if __name__ == '__main__':
    main()
