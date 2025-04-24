from abc import ABC, abstractmethod

"""
Abstract class for reading parameter arguments for a modeling task spec
"""


class ArgumentReader(ABC):
    """
    Method for parsing arguments provided some default_params container and a container of arguments to override default values
    """

    @abstractmethod
    def read_args(self, default_params, override_params):
        pass
