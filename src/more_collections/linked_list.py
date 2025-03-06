"""Linked List implementation in Python."""

from typing import Generic, Iterable, Iterator, Optional, TypeVar

from .node import Node

T = TypeVar("T")


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

    def append(self, value: T) -> Node[T]:
        node = Node(value)
        if self.head is None:
            self.head = node
            self.tail = node
        else:
            assert self.tail is not None
            self.tail.next = node
            node.prev = self.tail
            self.tail = node
        return node

    def prepend(self, value: T) -> Node[T]:
        node = Node(value)
        if self.head is None:
            self.head = node
            self.tail = node
        else:
            self.head.prev = node
            node.next = self.head
            self.head = node
        return node

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
