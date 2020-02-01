import os
import cv2
from datetime import datetime
import logging
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler

from .FPMatch import FPMatch


class Watcher:
    def __init__(self, inpath, outpath):
        self.observer = PollingObserver()
        self.inpath = inpath
        self.outpath = outpath

    def run(self):
        event_handler = Handler(self.outpath)
        self.observer.schedule(event_handler, self.inpath, recursive=True)
        self.observer.start()


class Handler(FileSystemEventHandler):
    def __init__(self, outpath):
        self.logger = logging.getLogger(__name__)
        self.outpath = outpath

    def on_any_event(self, event, **kwargs):
        if event.is_directory:
            return None
        elif event.event_type == 'created':
            self.logger.info(f'Received {event.event_type} event - {event.src_path}')
            st = datetime.now()
            image = cv2.imread(event.src_path, cv2.IMREAD_GRAYSCALE)
            FPMatch(image).dump(f'{self.outpath}/{os.path.basename(event.src_path)}')
            self.logger.info(f'Time elapsed: {datetime.now() - st}')
