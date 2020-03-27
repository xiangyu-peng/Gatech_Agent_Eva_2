import logging
def ini_log_level():

    logger = logging.getLogger('gameplay')
    if logger.handlers:
        logger.removeHandler('gameplay.log')
    # hdlr = logging.FileHandler('gameplay.log')
    # formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    # hdlr.setFormatter(formatter)
    # logger.addHandler(hdlr)
    # logger.setLevel(logging.DEBUG)

    return logger
def set_log_level():
    logger = logging.getLogger(__name__)
    # logger.setLevel(logging.DEBUG)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        hdlr = logging.FileHandler('gameplay.log', mode='a')
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)

    return logger

# logger = ini_log_level()
# logger = set_log_level()
# logger.debug('love')