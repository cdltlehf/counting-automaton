"""Linked List implementation in Python."""

from typing import Generic, Iterable, Iterator, Optional, TypeVar
import warnings

from .node import Node

T = TypeVar("T")


class LinkedList(Generic[T], Iterable[Node[T]]):
    """Doubly linked list"""

    def __init__(self) -> None:
        self._head: Optional[Node[T]] = None
        self._tail: Optional[Node[T]] = None

    def sanity_check(self) -> None:
        """Check if the linked list is sane"""
        if not __debug__:
            warnings.warn("Sanity checks are disabled", RuntimeWarning)
            return
        if self._head is None:
            assert self._tail is None
            return
        assert self._tail is not None
        assert self._head.prev is None
        assert self._tail.next is None
        current = self._head
        while current.next is not None:
            assert current.next.prev == current
            current = current.next
        assert current == self._tail

    @property
    def head(self) -> Optional[Node[T]]:
        return self._head

    @property
    def tail(self) -> Optional[Node[T]]:
        return self._tail

    def __iter__(self) -> Iterator[Node[T]]:
        current = self._head
        while current is not None:
            yield current
            current = current.next

    def is_empty(self) -> bool:
        return self._head is None

    def append_node(self, node: Node[T]) -> Node[T]:
        if self._head is None:
            self._head = node
            self._tail = node
        else:
            assert self._tail is not None, str(self)
            self._tail.next = node
            node.prev = self._tail
            self._tail = node
        if __debug__:
            LinkedList.sanity_check(self)
        return node

    def append(self, value: T) -> Node[T]:
        node = Node(value)
        return self.append_node(node)

    def prepend_node(self, node: Node[T]) -> Node[T]:
        if self._head is None:
            self._head = node
            self._tail = node
        else:
            self._head.prev = node
            node.next = self._head
            self._head = node
        if __debug__:
            LinkedList.sanity_check(self)
        return node

    def prepend(self, value: T) -> Node[T]:
        node = Node(value)
        return self.prepend_node(node)

    def remove(self, node: Node[T]) -> Node[T]:
        if node.prev is not None:
            node.prev.next = node.next
        if node.next is not None:
            node.next.prev = node.prev
        if node == self._head:
            self._head = node.next
        if node == self._tail:
            self._tail = node.prev
        if __debug__:
            LinkedList.sanity_check(self)
        return node

    def __str__(self) -> str:
        return " -> ".join(map(lambda node: str(node.value), self))

    def __repr__(self) -> str:
        return str(self)

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __copy__(self) -> "LinkedList[T]":
        new: LinkedList[T] = self.__class__()
        for node in self:
            new.append(node.value)
        return new

    def insert(self, node: Node[T], value: T) -> Node[T]:
        new_node = Node(value)
        self.insert_node(node, new_node)
        return new_node

    def insert_node(self, node: Node[T], new_node: Node[T]) -> Node[T]:
        new_node.prev = node
        new_node.next = node.next
        if node.next is not None:
            node.next.prev = new_node
        node.next = new_node
        if node == self._tail:
            self._tail = new_node
        if __debug__:
            self.sanity_check()
        return new_node
