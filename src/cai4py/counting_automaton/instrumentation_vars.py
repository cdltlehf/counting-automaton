op_name_to_count: dict[str, int] = {
    op_name: 0
    for op_name in [
        "INCREASE",
        "CLONE",
        "ADD_ONE",
        "ADD_ZERO",
        "REMOVE",
        "MERGE",
        "CHECK",
    ]
}
merge_set_sizes: list[tuple[int, float, int, float]] = []
clone_set_sizes: list[tuple[int, float]] = []
