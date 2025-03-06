"""_re.py"""

try:
    from re.compile import compile as _compile  # type: ignore
    from re.parser import State  # type: ignore
    from re.parser import SubPattern

except ImportError:
    from sre_compile import compile as _compile
    from sre_parse import State
    from sre_parse import SubPattern

__all__ = ["_compile", "State", "SubPattern"]
