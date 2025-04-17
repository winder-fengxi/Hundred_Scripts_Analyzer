import logging
import os


def get_logger(name=None, log_file=None, level=logging.INFO):
    """
    创建并返回一个 logger，用于输出到控制台和文件
    - name: logger 的名称
    - log_file: 如果提供，则同时将日志写入该文件
    - level: 日志级别
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加 Handler
    if logger.hasHandlers():
        logger.handlers.clear()

    # 控制台 Handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    # 文件 Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)

    return logger
