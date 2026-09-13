"""Structural application protocol for OpenNeuro.

This package defines the two structural contracts that connect any
application to the OpenNeuro runtime:

- ``ApplicationConnector``: implemented by each application.
- ``ApplicationGateway``: implemented by the runtime and handed to each
  connector's ``start()``.

Both are ``typing.Protocol`` classes: any object with the right shape
satisfies them, with no inheritance required. This module contains no
application-name literals and performs no I/O.
"""

from openneuro.protocol.connector import ApplicationConnector
from openneuro.protocol.gateway import ApplicationGateway

__all__ = ["ApplicationConnector", "ApplicationGateway"]
