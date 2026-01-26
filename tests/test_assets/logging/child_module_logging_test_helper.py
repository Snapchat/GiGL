from gigl.common.logger import Logger

logger = Logger()


def fn_write_to_parent_handler_from_child_module() -> None:
    logger.info("fn_write_to_parent_handler_from_child_module")


def fn_write_to_file_handler_from_child_module() -> None:
    # If `fn_write_to_file_handler_from_child_module` is called before `fn_write_to_parent_handler_from_child_module`
    # This logger will also capture the log above in fn_write_to_parent_handler_from_child_module
    # Why? Logs are captured at a module level i.e. tests.test_assets.logging.child_module_logging_test_helper
    # Why? See: https://docs.python.org/3/library/logging.html#logger-objects
    # tl;dr Loggers that are further down in the hierarchical list are children of loggers higher up in the list.
    # For example, given a logger with a name of foo, loggers with names of foo.bar, foo.bar.baz,
    # and foo.bam are all descendants of foo.
    #
    # This hierchichal structure becomes hard to manage when you introduce functions into the mix on top of module names
    # as you will need to essentially look at the stack of function calls every time rather than the module name for the
    # log name. This also means that if you if you were calling some.common.utility from two diferent modules; they would
    # have different log names. We probably don't want this.
    custom_file_logger = Logger(log_to_file=True)
    custom_file_logger.info("fn_write_to_file_handler_from_child_module")
