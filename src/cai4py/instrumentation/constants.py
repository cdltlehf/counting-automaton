OP_NAMES = sorted(
    [
        "INCREASE",
        "CLONE",
        "ADD_ONE",
        "ADD_ZERO",
        "REMOVE",
        "MERGE",
        "CHECK",
    ]
)
THROUGHPUT_THRES = 50 * 1e3  # Kilobytes per second

# Timeout configurations (in seconds)
AUTOMATON_CREATION_TIMEOUT = (
    30  # Increased from 10s for nested patterns with counters
)
MATCHING_TIMEOUT_BUFFER = 0.5  # Minimal buffer for faster ReDoS detection
