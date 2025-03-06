from typing import Iterable, Optional

from .node import Node


class SharedCounterSet:
    def __init__(self, colors: int) -> None:
        self.colors = colors
        self.offsets: dict[int, int] = {color: 0 for color in range(colors)}
        self.heads: dict[int, Optional[Node]] = {
            color: None for color in range(colors)
        }
        self.tails: dict[int, Optional[Node]] = {
            color: None for color in range(colors)
        }

    def add_one(self, color: int) -> None:
        if self.get_minimum(color) == 1:
            return

        offset = self.offsets[color]
        node = Node(0, offset - 1, self.colors)
        head = self.heads[color]

        if head is None:
            self.heads[color] = node
            self.tails[color] = node
            return

        head.prevs[color] = node
        node.nexts[color] = head

    def increase(self, color: int) -> None:
        self.offsets[color] += 1

    def get_maximum(self, color: int) -> Optional[int]:
        tail = self.tails[color]
        if tail is None:
            return None
        return tail.value

    def get_minimum(self, color: int) -> Optional[int]:
        head = self.tails[color]
        if head is None:
            return None
        return min(head.get_values(color))

    def _merge_into(self, color: int, other: int) -> None:
        raise NotImplementedError()

    def union(self, color: int, other: int) -> None:
        raise NotImplementedError()

    def iterate(self, color: int) -> Iterable[int]:
        node = self.heads[color]
        while True:
            if node is None:
                return
            for value in node.get_values(color):
                yield value
            node = node.nexts[color]

    def to_dict(self) -> dict[int, list[int]]:
        return {
            color: list(self.iterate(color)) for color in range(self.colors)
        }

    def print(self) -> None:
        for color in range(self.colors):
            print(f"{color}: {list(self.iterate(color))}")
