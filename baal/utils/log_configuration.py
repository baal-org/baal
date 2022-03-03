# flake8: noqa F841
"""
Note - This is a copied from the another internal repo.
Also inspired by https://gist.github.com/tboquet/588955d846c03de66b87c8bca0fb99cb
"""
import collections
import inspect
import os
import threading

import structlog
from structlog.dev import ConsoleRenderer
from structlog.processors import StackInfoRenderer, TimeStamper, format_exc_info
from structlog.stdlib import add_log_level


def set_logger_config():
    structlog.configure(
        processors=[
            structlog.stdlib.PositionalArgumentsFormatter(),
            StackInfoRenderer(),
            format_exc_info,
            structlog.processors.UnicodeDecoder(),
            TimeStamper(fmt="iso", utc=True),
            add_pid_thread,
            add_log_level,
            add_caller_info,
            order_keys,
            BetterConsoleRenderer(),
        ],
        context_class=structlog.threadlocal.wrap_dict(dict),
        cache_logger_on_first_use=True,
    )


def _foreground_color(color):
    return "\x1b[" + str(color) + "m"


# pylint: disable=W0612
def _level_styles():
    red = 31
    green = 32
    yellow = 33
    blue = 34
    purple = 35
    cyan = 36
    dark_gray = 90
    light_gray = 37
    light_red = 91
    light_green = 92
    light_yellow = 93
    light_blue = 94
    light_magenta = 95
    light_cyan = 96
    white = 97

    return {
        "critical": _foreground_color(red),
        "exception": _foreground_color(red),
        "error": _foreground_color(red),
        "warning": _foreground_color(yellow),
        "info": _foreground_color(green),
        "debug": _foreground_color(blue),
        "notset": _foreground_color(light_gray),
    }


def add_pid_thread(_, __, event_dict):
    pid = os.getpid()
    thread = threading.current_thread().getName()
    event_dict["pid_thread"] = f"{pid}-{thread}"
    return event_dict


def add_caller_info(logger, method_name, event_dict):
    # Typically skipped funcs: _add_caller_info, _process_event, _proxy_to_logger, _proxy_to_logger
    frame = inspect.currentframe()
    while frame:
        frame = frame.f_back
        module = frame.f_globals["__name__"]
        if module.startswith("structlog."):
            continue
        event_dict["module"] = module
        event_dict["lineno"] = frame.f_lineno
        event_dict["func"] = frame.f_code.co_name
        return event_dict


def order_keys(logger, method_name, event_dict):
    return collections.OrderedDict(
        sorted(event_dict.items(), key=lambda item: (item[0] != "event", item))
    )


class BetterConsoleRenderer:
    def __init__(self):
        self._worse_console_renderer = ConsoleRenderer(level_styles=_level_styles())

    def __call__(self, logger, log_method, event_dict):
        pid = event_dict.pop("pid_thread", None)
        module = event_dict.pop("module", None)
        func = event_dict.pop("func", None)
        lineno = event_dict.pop("lineno", None)
        if pid is None:
            return self._worse_console_renderer(logger, log_method, event_dict)
        mod_func_line = "{}:{}:{}".format(module, func, lineno).ljust(35)
        mod_func_line = "[{}] ".format(mod_func_line)
        pid_thread = "[{}] ".format(str(pid).ljust(16))
        return (
            pid_thread
            + mod_func_line
            + self._worse_console_renderer(logger, log_method, event_dict)
        )
