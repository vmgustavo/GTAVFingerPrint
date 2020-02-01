import logging.config as lconfig
import json


class Logger:
    def __init__(self):
        with open('logger.json', 'rt') as f:
            config = json.load(f)
        lconfig.dictConfig(config)
