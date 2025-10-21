from __future__ import annotations

from typing import Iterator


class Lineage:
    def __init__(self, clusters: list[int]) -> None:
        self.clusters = clusters

    def __len__(self) -> int:
        return len(self.clusters)

    def __repr__(self) -> str:
        return "Lineage" + str(self.clusters)

    def __iter__(self) -> Iterator[int]:
        yield from self.clusters
