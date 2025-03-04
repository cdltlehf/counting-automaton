from typing import Iterator, Optional


class Node():
    def __init__(self, index: int, value: int, colors: int) -> None:
        self.index = index
        self.value = value
        self.nexts: dict[int, Optional[Node]] = {
            color: None for color in range(colors)
        }
        self.prevs: dict[int, Optional[Node]] = {
            color: None for color in range(colors)
        }
        # TODO: offset ruins sortedness
        self.offsets: dict[int, set[int]] = {
            color: set() for color in range(colors)
        }

    def get_values(self, color: int) -> list[int]:
        return [self.value + offset for offset in self.offsets[color]]
