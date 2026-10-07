class Logger:
    def __init__(self, *args, **kwargs):
        pass


def rank_zero_experiment(fn):
    return fn
