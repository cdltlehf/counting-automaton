"""Additional collections classes."""

from typing import Generic, Iterable, Iterator, Optional, TypeVar
from weakref import proxy

T = TypeVar("T")


class Node(Generic[T]):
    next: "Node[T]"
    prev: "Node[T]"
    value: Optional[T]

    def __init__(self, value: Optional[T] = None) -> None:
        self.next = self
        self.prev = self
        self.value = value


class OrderedSet(Generic[T]):
    "Set the remembers the order elements were added"

    def __init__(self, iterable: Optional[Iterable[T]] = None) -> None:
        """Time complexity: O(n)"""
        root: Node[T] = Node()
        self._root = root
        self._map: dict[T, Node[T]] = {}
        if iterable is not None:
            for value in iterable:
                self.append(value)

    @property
    def first(self) -> Optional[T]:
        """Time complexity: O(1)"""
        return self._root.next.value

    @property
    def last(self) -> Optional[T]:
        """Time complexity: O(1)"""
        return self._root.prev.value

    @property
    def _last_node(self) -> Node[T]:
        return self._root.prev

    @property
    def _first_node(self) -> Node[T]:
        return self._root.next

    def _remove_duplicates(self, reverse: bool = False) -> None:
        """Time complexity: O(n). Remove duplicates from the set. If reverse is
        True, the last occurrence of a value is kept. Otherwise, the first
        occurrence of a value is kept.
        """

        seen: set[T] = set()
        root = self._root
        if reverse:
            curr = root.prev
            while curr is not root:
                assert curr.value is not None
                if curr.value in seen:
                    prev = curr.prev
                    prev.next = curr.next
                    curr.next.prev = prev
                else:
                    seen.add(curr.value)
                    self._map[curr.value] = curr
                curr = curr.prev
        else:
            curr = root.next
            while curr is not root:
                assert curr.value is not None
                if curr.value in seen:
                    prev = curr.prev
                    prev.next = curr.next
                    curr.next.prev = prev
                else:
                    seen.add(curr.value)
                    self._map[curr.value] = curr
                curr = curr.next

    def append_iterable(self, values: Iterable[T], at: Optional[T] = None) -> None:
        """Time complexity: O(n)"""
        if at is None:
            node = self._last_node
        elif at in self._map:
            node = self._map[at]
        else:
            raise KeyError(f"{at} not in set")

        for value in values:
            new = Node(value)
            self._map[value] = new
            self._append_node(new, node)
            node = new
        self._remove_duplicates(False)

    def append(self, value: T, at: Optional[T] = None) -> None:
        """Time complexity: O(1)"""
        self.append_iterable([value], at)

    def prepend_iterable(self, values: Iterable[T], at: Optional[T] = None) -> None:
        """Time complexity: O(1)"""
        if at is None:
            node = self._first_node.prev
        elif at in self._map:
            node = self._map[at]
        else:
            raise KeyError(f"{at} not in set")

        for value in values:
            new = Node(value)
            self._map[value] = new
            self._append_node(new, node)
            node = new
        self._remove_duplicates(False)

    def prepend(self, key: T, at: Optional[T] = None) -> None:
        """Time complexity: O(1)"""
        self.prepend_iterable([key], at)

    def substitute(self, old: T, new_values: Iterable[T]) -> None:
        """Time complexity: O(n)"""
        if old not in self._map:
            raise KeyError(f"{old} not in set")

        node = self._map.pop(old)
        node.prev.next = node.next
        node.next.prev = node.prev

        self.append_iterable(new_values, node.prev.value)

    def remove(self, value: T) -> None:
        """Time complexity: O(1)"""
        if value not in self._map:
            raise KeyError(f"{value} not in set")
        node = self._map.pop(value)
        node.prev.next = node.next
        node.next.prev = node.prev

    def __len__(self) -> int:
        """Time complexity: O(1)"""
        return len(self._map)

    def __iter__(self) -> Iterator[T]:
        """Time complexity: O(n)"""
        root = self._root
        curr = root.next
        while curr is not root:
            assert curr.value is not None
            yield curr.value
            curr = curr.next

    def __contains__(self, value: object) -> bool:
        """Time complexity: O(1)"""
        return value in self._map

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({list(self)})"

    def _append_node(self, new_node: Node[T], target: Node[T]) -> None:
        new_node.next = target.next
        new_node.prev = target
        target.next.prev = new_node
        target.next = new_node
