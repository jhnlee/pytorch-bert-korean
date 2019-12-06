import json
import torch
import logging
from pathlib import Path


formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')


def get_logger(name=None, level=logging.DEBUG):
    logger = logging.getLogger(name if name is not None else __name__)
    logger.handlers.clear()
    logger.setLevel(level)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    return logger


def acc(yhat, y):
    with torch.no_grad():
        acc = (yhat == y).float().mean()
    return acc


class SummaryManager:
    def __init__(self, model_dir):
        if not isinstance(model_dir, Path):
            model_dir = Path(model_dir)
        self._model_dir = model_dir
        self._summary = {}

    def save(self, filename):
        with open(self._model_dir / filename, mode='w') as io:
            json.dump(self._summary, io, indent=4)

    def load(self, filename):
        with open(self._model_dir / filename, mode='r') as io:
            metric = json.loads(io.read())
        self.update(metric)

    def update(self, summary):
        self._summary.update(summary)

    def reset(self):
        self._summary = {}

    @property
    def summary(self):
        return self._summary

