import os


def get_logger(logger_name: str = None):
    import logging
    import structlog
    from collections import OrderedDict

    if os.environ.get('ENV', 'dev') == 'dev':
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.ExceptionPrettyPrinter(),
            structlog.processors.TimeStamper('%H:%M'),
            structlog.dev.ConsoleRenderer(pad_event=40),
        ],
        context_class=OrderedDict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    log = structlog.get_logger(logger_name)
    return log
