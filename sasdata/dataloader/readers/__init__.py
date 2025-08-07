# Method to associate extensions to default readers
from .associations import get_fallback_readers, read_associations

__all__ = ["read_associations", "get_fallback_readers"]
