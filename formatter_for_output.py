"""
    Support classes for coloring the console output
"""

import logging
import sys

from config import FrameworkConfiguration


def format_console_output():
    """
    Format console with a common format and if selected with a colored output
    """
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='[%(levelname)s]\t(%(threadName)s) %(message)s', )
    logging.basicConfig(stream=sys.stdout, level=logging.ERROR, format='[%(levelname)s]\t(%(threadName)s) %(message)s', )
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(levelname)s]\t(%(threadName)s) %(message)s', )
    logging.basicConfig(stream=sys.stdout, level=logging.WARNING, format='[%(levelname)s]\t(%(threadName)s) %(message)s', )
    # Set colored output for console
    if FrameworkConfiguration.use_colored_output and FrameworkConfiguration.DEBUG is False:
        LOG = logging.getLogger()
        LOG.setLevel(logging.DEBUG)
        for handler in LOG.handlers:
            LOG.removeHandler(handler)
        LOG.addHandler(ColorHandler())


class _AnsiColorizer(object):
    """
    A colorizer is an object that loosely wraps around a stream, allowing
    callers to write text to the stream in a particular color.

    Colorizer classes must implement C{supported()} and C{write(text, color)}.
    """
    _colors = dict(black=30, red=31, green=32, yellow=33,
                   blue=34, magenta=35, cyan=36, white=37)

    def __init__(self, stream):
        self.stream = stream

    @classmethod
    def supported(cls, stream=sys.stdout):
        """
        A class method that returns True if the current platform supports
        coloring terminal output using this method. Returns False otherwise.
        """
        if not stream.isatty():
            return False  # auto color only on TTYs
        try:
            import curses
        except ImportError:
            return False
        else:
            try:
                try:
                    return curses.tigetnum("colors") > 2
                except curses.error:
                    curses.setupterm()
                    return curses.tigetnum("colors") > 2
            except:
                raise
                # guess false in case of error
                return False

    def write(self, text, color):
        """
        Write the given text to the stream in the given color.

        @param text: Text to be written to the stream.
        @param color: A string label for a color. e.g. 'red', 'white'.
        """
        color = self._colors[color]
        self.stream.write('\x1b[%s;1m%s\x1b[0m' % (color, text))


class ColorHandler(logging.StreamHandler):
    def __init__(self, stream=sys.stderr):
        super(ColorHandler, self).__init__(_AnsiColorizer(stream))

    def emit(self, record):
        msg_colors = {
            logging.DEBUG: "green",
            logging.INFO: "blue",
            logging.WARNING: "yellow",
            logging.ERROR: "red"
        }

        color = msg_colors.get(record.levelno, "green")
        self.stream.write(record.msg + "\n", color)
