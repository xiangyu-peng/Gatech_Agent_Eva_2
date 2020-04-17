import logging
import os
filename = 'gameplay_2'
def ini_log_level():


    logger = logging.getLogger(filename)
    if logger.handlers:
        logger.removeHandler(filename+'.log')
        # logging.Logger.manager.loggerDict.pop(__name__)
        # logger.handlers = []
        # logger.removeHandler(logger.handlers)

    return logger

def set_log_level():
    logger = logging.getLogger(__name__)
    # logger.setLevel(logging.CRITICAL)
    # if level == 'debug':
    #     logger.setLevel(logging.DEBUG)
    # if level == 'info':
    #     logger.setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    if os.path.exists('/media/becky/GNOME-p3/monopoly_simulator/'+ filename+'.log'):
        pass
    else:
        hdlr = logging.FileHandler(filename+'.log', mode='w')
        # formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        formatter = logging.Formatter('')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)

    if not logger.handlers:
        # print('add!!!')
        hdlr = logging.FileHandler(filename+'.log', mode='w')
        # formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        formatter = logging.Formatter('')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)
    return logger

class Logging_info():
    def __init__(self):
        self.level = 'debug'
    def set_level(self, level):
        self.level = level

# logger = ini_log_level()
# logger.debug('ini')

# logger_class = Logging_info()
#
# logger = set_log_level(logger_class.level)
# logger.debug('love')
# logger_class.set_level('info')
# logger = set_log_level(logger_class.level)
# logger.info('love')