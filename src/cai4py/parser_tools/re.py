"""_re.py"""

try:
    from re.compile import compile as _compile  # type: ignore
    from .parser import State  # type: ignore
    from .parser import SubPattern

except ImportError:
    from sre_compile import compile as _compile
    from .parser import State
    from .parser import SubPattern

__all__ = ["_compile", "State", "SubPattern"]
