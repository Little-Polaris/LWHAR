# import logging
#
#
# def miniLogger(start_time):
#     logger = logging.getLogger()
#     logger.setLevel(logging.INFO)  # 设置最低日志级别为 DEBUG
#
#     # 创建 formatter
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#
#     # 创建 console handler 并设置 level 为 DEBUG
#     ch = logging.StreamHandler()
#     ch.setLevel(logging.INFO)
#     ch.setFormatter(formatter)
#
#     # 创建 file handler 并设置 level 为 DEBUG
#     fh = logging.FileHandler(f'logs/{start_time}/log.txt', mode='w')  # 'w' 模式会覆盖已有文件，'a' 模式会追加
#     fh.setLevel(logging.INFO)
#     fh.setFormatter(formatter)
#
#     # 将 handler 添加到 logger
#     logger.addHandler(ch)
#     logger.addHandler(fh)
#     return logger

import logging
import multiprocessing
import sys
import traceback

from utils.after_finish import after_finish


class miniLogger:
    def __init__(self, start_time):
        """
        初始化 Logger 类。

        Args:
            name (str): Logger 的名称。 通常使用 __name__。
            level (int): 日志级别 (例如 logging.DEBUG, logging.INFO)。 默认为 logging.DEBUG。
            log_file (str, optional): 日志文件的路径。 如果为 None，则只输出到控制台。
        """
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # 创建 console handler
        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.INFO)
        self.ch.setFormatter(self.formatter)
        self.logger.addHandler(self.ch)

        # 创建 file handler (如果指定了 log_file)
        self.fh = logging.FileHandler(f'logs/{start_time}/log.txt', mode='w')  # 追加模式
        self.fh.setLevel(logging.INFO)
        self.fh.setFormatter(self.formatter)
        self.logger.addHandler(self.fh)

        # 设置自定义的异常钩子 (可选)
        self._set_excepthook()

    def _set_excepthook(self):
        """
        设置自定义的异常钩子，用于记录未捕获的异常。
        """
        def my_excepthook(exc_type, exc_value, exc_traceback):
            error_message = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            self.logger.critical(f"未捕获的异常：\n{error_message}")
            shutdown_process = multiprocessing.Process(target=after_finish, args=(True,))
            shutdown_process.start()
            sys.__excepthook__(exc_type, exc_value, exc_traceback)  # 调用默认的钩子

        sys.excepthook = my_excepthook

    def debug(self, message, *args, **kwargs):
        self.logger.debug(message, *args, **kwargs)

    def info(self, message, *args, **kwargs):
        self.logger.info(message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        self.logger.warning(message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        self.logger.error(message, *args, **kwargs)

    def critical(self, message, *args, **kwargs):
        self.logger.critical(message, *args, **kwargs)

    def exception(self, message, *args, **kwargs):
        self.logger.exception(message, *args, **kwargs)
