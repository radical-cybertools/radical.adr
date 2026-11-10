"""Reading annotations without evaluating them, on Python 3.10 through 3.14.

From Python 3.14 (PEP 649/749) annotations are evaluated lazily: class annotations
are not in the class ``__dict__`` unless the module uses ``from __future__ import
annotations``, and ``inspect.signature`` evaluates them by default, so a name that
is only imported under ``TYPE_CHECKING`` raises ``NameError``.
"""

from __future__ import annotations

import inspect
import sys
from typing import Any, Callable

if sys.version_info >= (3, 14):
    import annotationlib
else:
    annotationlib = None  # type: ignore[assignment]


def own_annotation_names(klass: type) -> list[str]:
    """Names of the annotations declared on ``klass`` itself (not inherited), unevaluated."""
    if annotationlib is not None:
        return list(annotationlib.get_annotations(klass, format=annotationlib.Format.STRING))
    return list(vars(klass).get("__annotations__", {}))


def signature(fn: Callable[..., Any]) -> inspect.Signature:
    """``inspect.signature`` that leaves an unresolvable annotation as a ``ForwardRef``."""
    if annotationlib is not None:
        return inspect.signature(fn, annotation_format=annotationlib.Format.FORWARDREF)
    return inspect.signature(fn)
