import sys 
import os
from src.utils.logging import logger


class claim_exception(Exception):
    def __init__(self, error_message, error_details:sys):
        self.error_message = error_message
        _,_,exc_tb = error_details.exc_info()

        self.lineno = exc_tb.tb_lineno
        self.file_name=exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return "Error occured in Python script name [{0}] line number [{1}] error message"
        self.file_name, self.lineno, str(self.error_message)

if __name__ == "__main__":
    try:
        logger.logging.INFO("Enter try block")
        a = 1/2
        print("Division by zero, not printable",a )
    except Exception as e:
        raise claim_exception(e,sys)
    
    finally:
        print("Run Complete")