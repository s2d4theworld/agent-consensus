# agent-consensus — exception types shared by all modules.
# Messages are user-facing: callers print them verbatim when a run aborts.

class ConsensusError(Exception):
    """Base class for all expected, user-facing failures."""

class TransportError(ConsensusError):
    """HTTP/transport-level failure (bad response, error envelope, missing key)."""

class ServerUnreachable(TransportError):
    """A configured server or endpoint cannot be reached at all."""
