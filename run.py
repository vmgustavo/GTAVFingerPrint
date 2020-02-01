import json
import os
import logging

from FPMatch import Watcher
import LoggerClass

LoggerClass.Logger()
logger = logging.getLogger(__name__)


def run():
    with open('config.json', 'r') as f:
        config = json.load(f)

    if not os.path.exists(config['input']):
        logger.info(f'Creating {config["input"]}')
        os.mkdir(config['input'])
    if not os.path.exists(config['output']):
        logger.info(f'Creating {config["output"]}')
        os.mkdir(config['output'])

    w = Watcher(inpath=config['input'], outpath=config['output'])
    logger.info(f'Monitoring {config["input"]}')
    w.run()


if __name__ == '__main__':
    run()
