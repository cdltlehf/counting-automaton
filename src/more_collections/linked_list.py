"""Linked List implementation in Python."""

import logging
from typing import Any, Callable, Generic, Iterable, Iterator, Optional, Protocol, TypeVar

from .node import Node


class Comparable(Protocol):
    def __lt__(self, other: Any) -> bool:
        pass


T = TypeVar("T", bound=Comparable)


class LinkedList(Generic[T], Iterable[Node[T]]):
    """Doubly linked list"""

    def __init__(self) -> None:
        self.head: Optional[Node[T]] = None
        self.tail: Optional[Node[T]] = None

    def __iter__(self) -> Iterator[Node[T]]:
        current = self.head
        while current is not None:
            yield current
            current = current.next

    def is_empty(self) -> bool:
        return self.head is None

    def append_node(self, node: Node[T]) -> Node[T]:
        if self.head is None:
            self.head = node
            self.tail = node
        else:
            assert self.tail is not None
            self.tail.next = node
            node.prev = self.tail
            self.tail = node
        return node

    def append(self, value: T) -> Node[T]:
        node = Node(value)
        return self.append_node(node)

    def prepend_node(self, node: Node[T]) -> Node[T]:
        if self.head is None:
            self.head = node
            self.tail = node
        else:
            self.head.prev = node
            node.next = self.head
            self.head = node
        return node

    def prepend(self, value: T) -> Node[T]:
        node = Node(value)
        return self.prepend_node(node)

    def remove(self, node: Node[T]) -> Node[T]:
        if node.prev is not None:
            node.prev.next = node.next
        if node.next is not None:
            node.next.prev = node.prev
        if node == self.head:
            self.head = node.next
        if node == self.tail:
            self.tail = node.prev
        return node

    def __str__(self) -> str:
        return " -> ".join(map(lambda node: str(node.value), self))

    def __repr__(self) -> str:
        return str(self)

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __copy__(self) -> "LinkedList[T]":
        new: LinkedList[T] = LinkedList()
        for node in self:
            new.append(node.value)
        return new

    def insert(self, node: Node[T], value: T) -> Node[T]:
        new_node = Node(value)
        new_node.prev = node
        new_node.next = node.next
        if node.next is not None:
            node.next.prev = new_node
        node.next = new_node
        return new_node

    def merge(
        self,
        other: "LinkedList[T]",
        key: Callable[[T, T], bool] = lambda a, b: a < b,
    ) -> "LinkedList[T]":
        if other.head is None:
            return self
        if self.head is None:
            self.head = other.head
            self.tail = other.tail
            return self

        assert self.tail is not None
        assert other.tail is not None

        self_node: Optional[Node[T]] = self.head
        other_node: Optional[Node[T]] = other.head
        while other_node is not None:
            _other_node = other_node
            other_node = other_node.next

            while self_node is not None and key(
                self_node.value, _other_node.value
            ):
                self_node = self_node.next

            # self_node.prev.value < _other_node.value <= self_node.value

            if self_node is None:
                self.append_node(_other_node)
                continue

            if self_node.value == _other_node.value:
                continue

            assert key(_other_node.value, self_node.value)

            if self_node.prev is None:
                self.prepend_node(_other_node)
                continue

            assert key(self_node.prev.value, _other_node.value)
            self.insert(self_node.prev, _other_node.value)
        return self
