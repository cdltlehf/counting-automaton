"""Node class for doubly linked list."""

from typing import Generic, Optional, TypeVar

T = TypeVar("T")


class Node(Generic[T]):
    next: Optional["Node[T]"]
    prev: Optional["Node[T]"]
    value: T

    def __init__(self, value: T) -> None:
        self.next = None
        self.prev = None
        self.value = value
