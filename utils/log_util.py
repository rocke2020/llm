import logging, os
import functools
import time
import contextlib


fmt = '%(asctime)s %(filename)s %(lineno)d: %(message)s'
datefmt = '%y-%m-%d %H:%M:%S'


def get_logger(name=None, log_file=None, log_level=logging.INFO):
    """ concise log """
    logger = logging.getLogger(name)
    logging.basicConfig(format=fmt, datefmt=datefmt)
    if log_file is not None:
        log_file_folder = os.path.split(log_file)[0]
        if log_file_folder:
            os.makedirs(log_file_folder, exist_ok=True)
        fh = logging.FileHandler(log_file, 'w', encoding='utf-8')
        fh.setFormatter(logging.Formatter(fmt, datefmt))
        logger.addHandler(fh)
    logger.setLevel(log_level)
    return logger

logger = get_logger()


def log_df_basic_info(df, comments=''):
    if comments:
        logger.info(f'comments {comments}')
    logger.info(f'df.shape {df.shape}')
    logger.info(f'df.columns {df.columns.to_list()}')
    logger.info(f'df.head()\n{df.head()}')
    logger.info(f'df.tail()\n{df.tail()}')


def func_time_print(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        t0 = time.time()
        res = func(*args, **kw)
        _total_seconds = time.time() - t0
        total_seconds = int(_total_seconds)
        hours = total_seconds // 3600
        total_seconds = total_seconds % 3600
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        print(f'call {func.__name__}() uses hours:mm:ss {hours}:{minutes}:{seconds}')
        print(f'call {func.__name__}() uses total seconds {_total_seconds:.3f}')
        return res
    return wrapper


@contextlib.contextmanager
def timing(msg: str):
  logging.info('Started %s', msg)
  tic = time.time()
  yield
  toc = time.time()
  total_seconds = toc - tic
  hours = total_seconds / 3600
  logging.info('Finished %s in %.3f seconds, i.e. %.3f hours', msg, total_seconds, hours)


if __name__ == "__main__":
    with timing('test'):
        a = 1
        for i in range(1000):
            a = 1
            
    pass
