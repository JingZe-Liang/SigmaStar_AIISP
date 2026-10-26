"""Minimal registry shim for the official AP-BSN network."""


def regist_model(cls):
    return cls


from .APBSN import APBSN
