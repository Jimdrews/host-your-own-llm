"""Errors shown to the user as a message, without a traceback."""


class CliError(Exception):
    pass


class ModelError(Exception):
    """A model can't be served from the model store."""
