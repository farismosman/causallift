import json, logging.config
from pathlib import Path

root_dir = str(Path(__file__).resolve().parent.parent.parent.parent)


class Loggers:
    def __init__(self, config_file=None):
        self.config_file = config_file
        self.log_level = {
            0: 'ERROR',
            1: 'WARN',
            2: 'INFO',
            3: 'DEBUG'
        }

    def config(self, verbose):
        with open('%s/conf/base/logging.json'%(root_dir)) as jsonfile:
            log_config = json.load(jsonfile)

        log_config['loggers']['causallift']['level'] = self.log_level[verbose]

        return log_config

    def setup(self, verbose):
        _config = self.config(verbose)
        logging.config.dictConfig(_config)