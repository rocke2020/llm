import logging, os
from pandas import DataFrame


_nameToLevel = {
#    'CRITICAL': logging.CRITICAL,
#    'FATAL': logging.FATAL,  # FATAL = CRITICAL
#    'ERROR': logging.ERROR,
#    'WARN': logging.WARNING,
   'INFO': logging.INFO,
   'DEBUG': logging.DEBUG,
}
fmt = '%(asctime)s %(filename)s %(lineno)d: %(message)s'
datefmt = '%y-%m-%d %H:%M:%S'


def get_logger(name=None, log_file=None, log_level=logging.DEBUG, log_level_name=''):
    """ default log level DEBUG """
    logger = logging.getLogger(name)
    logging.basicConfig(format=fmt, datefmt=datefmt)
    if log_file is not None:
        log_file_folder = os.path.split(log_file)[0]
        if log_file_folder:
            os.makedirs(log_file_folder, exist_ok=True)
        fh = logging.FileHandler(log_file, 'w', encoding='utf-8')
        fh.setFormatter(logging.Formatter(fmt, datefmt))
        logger.addHandler(fh)
    if log_level_name in _nameToLevel:
        log_level = _nameToLevel[log_level_name]
    logger.setLevel(log_level)
    return logger


def log_df_basic_info(df: DataFrame, log_func=None, comments='', full_info=False):
    """ For large df, describe() use much more time! """
    if log_func is None:
        log_func = logger
    if comments:
        log_func.info(f'comments {comments}')
    log_func.info(f'df.shape {df.shape}')
    columns = df.columns.to_list()
    if full_info:
        log_func.info(f'df.columns {columns}')
        log_func.info(f'df.head()\n{df.head()}')
        log_func.info(f'df.tail()\n{df.tail()}')
        log_func.info(f'df.describe()\n{df.describe()}')
    else:
        log_func.info(f'df.columns [:8] {columns[:8]}')
        log_func.info(f'df.columns [-5:] {columns[-5:]}')
        log_func.info(f'df[columns[:8]].head() {df[columns[:8]].head()}')
        log_func.info(f'df[columns[-5:]].tail() {df[columns[-5:]].tail()}')
        if len(df) <= 200_000:
            log_func.info(f'df[columns[:8]].describe()\n{df[columns[:8]].describe()}')


logger = get_logger()
