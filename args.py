import argparse
from calendar import c
from dataclasses import dataclass

from typing_extensions import Self


def config_args(config):
    config.parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    values = config.__dict__

    fields = values["__annotations__"]

    for (name, type) in fields.items():
        ps = {}

        default_value = values[name] if name in values else None

        if default_value:
            ps["default"] = default_value

        if type == bool:
            ps["action"] = "store_true"
        else:
            ps["type"] = type

        config.parser.add_argument(f"--{name}", **ps)

    def __init__(self):
        new_args = config.parser.parse_args()

        for (name, value) in new_args.__dict__.items():
            print(f"{name} = {value}")
            self.__dict__[name] = value

    config.__init__ = __init__

    return config
