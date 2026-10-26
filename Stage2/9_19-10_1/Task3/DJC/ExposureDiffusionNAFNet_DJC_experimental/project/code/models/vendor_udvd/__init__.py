"""Minimal registry shim for the vendored UDVD model classes."""


def register_model(_name):
    return lambda cls: cls


from .udvd import BlindVideoNet
