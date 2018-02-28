#!/usr/bin/env python
# -*- coding: utf-8 -*-

from optparse import OptionParser
import datetime
from feature_extractor import FeatureExtraction
from configurator import Configurator
from sensespotting_classifier import SenseSpottingClassifier
from psd_classifier import PSDClassifier

class ConfigException(Exception):
    pass

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-c", "--config", dest="config_file", default=None,
                      help='Give path to a configuration file (containing corpus information, model information , etc.)')
    parser.add_option("-u", "--use_existing", dest="use_existing", default='1',
                      help='DEVELOPMENT OPTION (not needed for final script). Values: 1 or 0.'
                           ' This option is only relevant if not all paths (REQUIRED and OPTIONAL) are provided '
                           'with config file. It defines if you want to re-use already created files or want to '
                           'guarantee that all files are created again. E.g. needed if you change the corpus data, '
                           'but you might want to use it if you work with the same data and run the script several times.')
    parser.add_option('-v', '--verbose', dest="verbose", default='1',
                      help="Show print statements or ignore them (Default: On)")

    (options, args) = parser.parse_args()
    reuse = int(options.use_existing)
    verbosity = int(options.verbose)

    if reuse not in [1, 0]:
        raise ConfigException('Value of <use_existing> must be 1 (True) or 0 (False)!')

    if verbosity not in [1, 0]:
        raise ConfigException('Value of <verbose> must be 1 (True) or 0 (False)!')

    if verbosity:
        print('\nSTART program ({})'.format(datetime.datetime.now()))

    psd = PSDClassifier(config_file=options.config_file, use_existing=reuse, verbose=verbosity)
    psd.run()

    clf = SenseSpottingClassifier(config_file=options.config_file, use_existing=reuse, verbose=verbosity)
    clf.experiment_name = 'sensespotting'
    clf.run()
    # clf.perform_ablation_study()

    if verbosity:
        print('\nFINISHED program! ({})'.format(datetime.datetime.now()))




