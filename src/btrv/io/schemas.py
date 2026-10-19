"""JSON-schema access and validation helpers."""
from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any

import jsonschema

from .errors import ImportValidationError


@lru_cache(maxsize=None)
def load_schema(name: str) -> dict[str, Any]:
    with resources.files("btrv.schemas").joinpath(name).open("r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=None)
def _validator(name: str) -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator(load_schema(name))


def validate(obj: Any, schema_name: str, context: str = "") -> None:
    errors = sorted(_validator(schema_name).iter_errors(obj), key=lambda e: list(e.path))
    if errors:
        e = errors[0]
        where = "/".join(str(p) for p in e.path) or "<root>"
        raise ImportValidationError(f"{context}{'; ' if context else ''}schema {schema_name}: at '{where}': {e.message}")
