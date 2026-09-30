"""Offline composition of model call sites with kernel-owned config selectors."""

from __future__ import annotations

import importlib
from typing import Any

from .dense_launch_plan import materialize_static_launch_plan
from .physical import PAGE_SIZE


def select_static_launches(
    contracts: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
    *,
    modules: dict[str, object] | None = None,
) -> list[dict[str, Any]]:
    """Materialize one model inventory using colocated kernel selectors."""

    loaded = {} if modules is None else dict(modules)
    selectors = {}
    for contract in contracts:
        module_name = contract["module"]
        module = loaded.get(module_name)
        if module is None:
            module = importlib.import_module(f"triton_kernels.{module_name}")
            loaded[module_name] = module
        selector = getattr(module, "select_config", None)
        if not callable(selector):
            raise RuntimeError(
                f"kernel module {module_name!r} has no colocated select_config"
            )
        selectors[contract["wrapper"]] = selector

    for module_name, module in loaded.items():
        compiled_page_size = getattr(module, "PAGE_SIZE", None)
        if compiled_page_size is not None and compiled_page_size != PAGE_SIZE:
            raise RuntimeError(
                f"{module_name} PAGE_SIZE differs from Engine PAGE_SIZE={PAGE_SIZE}"
            )
    return materialize_static_launch_plan(inventory, selectors)


__all__ = ["select_static_launches"]
