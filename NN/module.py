import re

class Module:
    def __init__(self, name=None) -> None:
        name = camel_to_snake(type(self).__name__)
        