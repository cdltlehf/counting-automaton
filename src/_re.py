try:
    from re.compile import compile as _compile
    from re.parser import SubPattern, State

except ImportError:
    from sre_compile import compile as _compile
    from sre_parse import SubPattern, State
