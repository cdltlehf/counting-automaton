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

THROUGHPUT_THRES = 50 * 1e-3  # Kilobytes per second (i.e. 50 B/s)
