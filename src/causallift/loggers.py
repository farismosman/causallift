import json, logging
from pathlib import Path


LOG_LEVEL = {
    0: 'ERROR',
    1: 'WARN',
    2: 'INFO',
    3: 'DEBUG'
}

root_dir = str(Path(__file__).resolve().parent.parent.parent)

def config(verbose):
    with open('%s/conf/base/logging.json'%(root_dir)) as jsonfile:
        log_config = json.load(jsonfile)

    log_config['loggers']['causallift']['level'] = LOG_LEVEL[verbose]

    return log_config


def setup(verbose):
    _config = config(verbose)
    logging.config.dictConfig(_config)