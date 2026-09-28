import sys
import traceback

from src.logger import logging


def error_message_detail(error, error_detail=sys):
    traceback_info = error_detail.exc_info()[2]
    if traceback_info is None:
        return str(error)
    frame = traceback.extract_tb(traceback_info)[-1]
    return f"{frame.filename}:{frame.lineno}: {error}"

class CustomException(Exception):
    def __init__(self, error_message, error_detail=sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)
        logging.error("%s", self.error_message)

    def __str__(self):
        return self.error_message
        



        
