"""Ordered set implementation"""

from typing import Generic, Iterable, Iterator, Optional, TypeVar

from .linked_list import LinkedList
from .node import Node

T = TypeVar("T")


class OrderedSet(Generic[T]):
    "Set the remembers the order elements were added"

    def __init__(self, iterable: Optional[Iterable[T]] = None) -> None:
        """Time complexity: O(n)"""
        self.list: LinkedList[T] = LinkedList()
        self.node: dict[T, Node[T]] = {}

        if iterable is not None:
            for value in iterable:
                self.append(value)

    @property
    def first(self) -> Optional[T]:
        """Time complexity: O(1)"""
        if self.list.head is None:
            return None
        return self.list.head.value

    @property
    def last(self) -> Optional[T]:
        """Time complexity: O(1)"""
        if self.list.tail is None:
            return None
        return self.list.tail.value

    def _remove_duplicates(self, reverse: bool = False) -> None:
        """Time complexity: O(n). Remove duplicates from the set. If reverse is
        True, the last occurrence of a value is kept. Otherwise, the first
        occurrence of a value is kept.
        """

        seen: set[T] = set()
        if reverse:
            node = self.list.tail
            while node is not None:
                if node.value in seen:
                    self.list.remove(node)
                else:
                    seen.add(node.value)
                    self.node[node.value] = node
                node = node.prev
        else:
            node = self.list.head
            while node is not None:
                if node.value in seen:
                    self.list.remove(node)
                else:
                    seen.add(node.value)
                    self.node[node.value] = node
                node = node.next

    def append_iterable(
        self, values: Iterable[T], at: Optional[T] = None
    ) -> None:
        """Time complexity: O(n)"""
        if at is None:
            node = self.list.tail
        elif at in self.node:
            node = self.node[at]
        else:
            raise KeyError(f"{at} not in set")

        for value in values:
            if node is None:
                node = self.list.append(value)
            else:
                node = self.list.insert(node, value)
            self.node[value] = node
        self._remove_duplicates(False)

    def append(self, value: T, at: Optional[T] = None) -> None:
        """Time complexity: O(1)"""
        self.append_iterable([value], at)

    def prepend_iterable(
        self, values: Iterable[T], at: Optional[T] = None
    ) -> None:
        """Time complexity: O(1)"""
        if at is None:
            node = self.list.head
        elif at in self.node:
            node = self.node[at]
        else:
            raise KeyError(f"{at} not in set")

        for value in values:
            if node is None:
                node = self.list.prepend(value)
            else:
                if node.prev is None:
                    node = self.list.prepend(value)
                else:
                    node = self.list.insert(node.prev, value)
            self.node[value] = node
        self._remove_duplicates(False)

    def prepend(self, key: T, at: Optional[T] = None) -> None:
        """Time complexity: O(1)"""
        self.prepend_iterable([key], at)

    def substitute(self, old: T, new_values: Iterable[T]) -> None:
        """Time complexity: O(n)"""
        if old not in self.node:
            raise KeyError(f"{old} not in set")

        node = self.node.pop(old)
        at = node.prev.value if node.prev is not None else None

        self.list.remove(node)
        self.append_iterable(new_values, at)

    def remove(self, value: T) -> None:
        """Time complexity: O(1)"""
        if value not in self.node:
            raise KeyError(f"{value} not in set")
        node = self.node.pop(value)
        self.list.remove(node)

    def __len__(self) -> int:
        """Time complexity: O(1)"""
        return len(self.node)

    def __iter__(self) -> Iterator[T]:
        """Time complexity: O(n)"""
        for node in self.list:
            yield node.value

    def __contains__(self, value: object) -> bool:
        """Time complexity: O(1)"""
        return value in self.node

    def __str__(self) -> str:
        return f"OrderedSet({', '.join(map(str, self))})"
