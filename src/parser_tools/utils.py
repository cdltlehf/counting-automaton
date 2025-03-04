def is_nullable(tree: SubPattern) -> bool:
    if has_extended_features(tree):
        raise ValueError("Pattern has extended features")

    def f(
        x: Optional[tuple[Constant, Any]], ys: Iterable[bool]
    ) -> bool:
        if x is None:
            return all(ys)

        opcode, operand = x
        if opcode in PREDICATE_OPCODES:
            return False
        elif opcode is AT:
            return True
        elif opcode is BRANCH:
            return any(ys)
        elif opcode in REPEAT_OPCODES:
            return operand[0] == 0 or all(ys)
        elif opcode in {ATOMIC_GROUP, SUBPATTERN}:
            return all(ys)
        else:
            assert False, f"Unknown opcode: {opcode}"

    return fold(f, tree)


def is_problematic(tree: SubPattern) -> bool:
    def f(
        x: Optional[tuple[Constant, Any]], ys: Iterable[bool]
    ) -> bool:
        if x is None:
            return any(ys)
        opcode, operand = x
        if opcode in REPEAT_OPCODES:
            _, n = operand
            if n is MAXREPEAT:
                return is_nullable(tree)
        return any(ys)

    return fold(f, tree)


def has_extended_features(tree: SubPattern) -> bool:
    return any(opcode in EXTENDED_OPCODES for opcode, _ in dfs(tree))


def is_literal_sequence(tree: SubPattern) -> bool:
    return all(opcode is LITERAL for opcode, _ in dfs(tree))


def is_anchored_literal_sequence(tree: SubPattern) -> bool:
    return all(
        opcode in {LITERAL, AT} | SUBPATTERN_OPCODES for opcode, _ in dfs(tree)
    )


def is_anchored_predicate_sequence(tree: SubPattern) -> bool:
    return all(
        opcode in {PREDICATE_OPCODES, AT} | SUBPATTERN_OPCODES
        for opcode, _ in dfs(tree)
    )


def is_finite_pattern(tree: SubPattern) -> bool:
    for opcode, operand in dfs(tree):
        if opcode in REPEAT_OPCODES:
            _, n = operand
            if n is MAXREPEAT:
                return False
    return True
