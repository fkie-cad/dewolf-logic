"""Module defining decorators used in the simplifier."""
from functools import wraps


def dirty(func):
    """Wrap the method of an operation to indicate that the dirty flag should be set."""

    @wraps(func)
    def set_dirty(self, *args, **kwargs):
        """Set the dirty flag of the Operation object."""
        self._dirty = True
        return func(self, *args, **kwargs)

    return set_dirty


def clean(func):
    """Wrap the method of an operation to indicate that the dirty flag should be cleared."""

    @wraps(func)
    def set_clean(self, *args, **kwargs):
        """Check and clear the dirty flag of the Operation object."""
        if not self._dirty:
            return
        value = func(self, *args, **kwargs)
        self._dirty = False
        return value

    return set_clean
