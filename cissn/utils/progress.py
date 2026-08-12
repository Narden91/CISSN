"""Consistent terminal progress reporting for experiment loops."""
from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

from tqdm.auto import tqdm


Item = TypeVar("Item")


def track(
    items: Iterable[Item],
    *,
    description: str,
    total: int | None = None,
    enabled: bool = True,
) -> Iterable[Item]:
    """Return a compact progress bar, or the original iterable when disabled."""
    if not enabled:
        return items
    return tqdm(
        items,
        desc=description,
        total=total,
        unit="batch",
        dynamic_ncols=True,
        leave=False,
        mininterval=0.2,
    )
