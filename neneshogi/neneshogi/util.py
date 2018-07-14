import sys
import os
import yaml


def yaml_load(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f)


def yaml_dump(obj: object, path: str):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(obj, f, default_flow_style=False)
