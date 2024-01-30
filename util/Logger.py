import time

from util.PrintColors import PrintColors


def printc(text, color=PrintColors.DEFAULT):
    """
    Function for printing a string in a color

    :param text: string which will be printed
    :param color: color of the text (default: PrintColors.DEFAULT)
    """
    print(f'{color}{text}{PrintColors.DEFAULT}')


class Logger:
    def __init__(self, logging):
        self.logging = logging
        self.start_time = time.time()
        self.last_time = self.start_time

    def log(self, message, color=PrintColors.DEFAULT, exec_time=True, current_time=True):
        """
        Function for logging a message with timestamp

        :param message: message for log entry
        :param color: color of the message (default: PrintColors.DEFAULT)
        :param exec_time: flag if execution time should be displayed (default: True)
        :param current_time: flag if current time should be displayed (default: True)
        """

        lines = str(message).split('\n')

        if self.logging:
            for line in lines:
                if exec_time and current_time:
                    printc(
                        f'[{time.strftime("%H:%M:%S", time.localtime())}]: {line}  -  Execution took {time.time() - self.last_time:.2f}s. Total: {time.time() - self.start_time:.2f}s',
                        color=color)
                elif exec_time and not current_time:
                    printc(
                        f'{line}  -  Execution took {time.time() - self.last_time:.2f}s. Total: {time.time() - self.start_time:.2f}s',
                        color=color)
                elif not exec_time and current_time:
                    printc(f'[{time.strftime("%H:%M:%S", time.localtime())}]: {line}', color=color)
                elif not exec_time and not current_time:
                    printc(f'{line}', color=color)

        self.last_time = time.time()

    def get_duration(self):
        return time.time() - self.last_time

