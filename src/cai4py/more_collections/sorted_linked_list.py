"""Sorted Doubly linked list"""

from typing import Any, Callable, Optional, Protocol, TypeVar
import warnings

from .linked_list import LinkedList
from .node import Node


class Comparable(Protocol):
    def __lt__(self, other: Any) -> bool:
        pass


T = TypeVar("T", bound=Comparable)


class SortedLinkedList(LinkedList[T]):
    """Sorted Doubly linked list"""

    def __init__(
        self,
        key: Callable[[T, T], bool] = lambda a, b: a < b,
    ) -> None:
        super().__init__()
        self.key = key

    def merge(
        self,
        other: "LinkedList[T]",
    ) -> "LinkedList[T]":
        if other.head is None:
            return self
        if self.head is None:
            self._head = other.head
            self._tail = other.tail
            return self

        assert self.tail is not None
        assert other.tail is not None

        self_node: Optional[Node[T]] = self.head
        other_node: Optional[Node[T]] = other.head
        while other_node is not None:
            current_other_node = other_node
            other_node = other_node.next

            # Find the first self-node that is greater than or equal to the
            # current other-node
            while self_node is not None and self.key(
                self_node.value, current_other_node.value
            ):
                self_node = self_node.next

            # If there is no such self-node, append the current other-node to
            # the end of the list
            if self_node is None:
                self.append_node(current_other_node)
                continue

            # If the current other-node is equal to the current self-node, skip
            if self_node.value == current_other_node.value:
                continue

            assert self.key(current_other_node.value, self_node.value)

            if self_node.prev is None:
                self.prepend_node(current_other_node)
                continue

            assert self.key(self_node.prev.value, current_other_node.value), (
                self_node.prev.value,
                current_other_node.value,
            )
            self.insert_node(self_node.prev, current_other_node)

        return self

    def __copy__(self) -> "SortedLinkedList[T]":
        new: SortedLinkedList[T] = SortedLinkedList(self.key)
        for node in self:
            new.append(node.value)
        if __debug__:
            SortedLinkedList.sanity_check(new)
        return new

    def sanity_check(self) -> None:
        if not __debug__:
            warnings.warn("Sanity checks are disabled", RuntimeWarning)
            return
        super().sanity_check()
        if self.head is None:
            return
        current = self.head
        while current.next is not None:
            assert self.key(current.value, current.next.value)
            current = current.next
        assert current == self.tail
