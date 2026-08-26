"""Electrode data model with dict-like access for backward compatibility."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


class Electrode(dict):
    """Container for electrode properties with attribute and dict-style access."""

    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        shaft: str,
        *,
        view: Optional[int] = None,
        name: Optional[str] = None,
        visible: bool = True,
        detected: bool = False,
        space: str = "unregistered",
        tissue: str = "unknown",
        slice_axial: Optional[int] = None,
        slice_coronal: Optional[int] = None,
        slice_sagittal: Optional[int] = None,
        **extra: Any,
    ) -> None:
        if slice_axial is None:
            slice_axial = int(round(z))
        if slice_coronal is None:
            slice_coronal = int(round(y))
        if slice_sagittal is None:
            slice_sagittal = int(round(x))

        data: Dict[str, Any] = {
            "view": view,
            "name": name,
            "x": x,
            "y": y,
            "z": z,
            "shaft": shaft,
            "3d_pos": (x, y, z),
            "slice_axial": slice_axial,
            "slice_coronal": slice_coronal,
            "slice_sagittal": slice_sagittal,
            "visible": visible,
            "detected": detected,
            "space": space,
            "tissue": tissue,
        }
        data.update(extra)
        super().__init__(data)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        self[name] = value

    @property
    def location(self) -> Tuple[float, float, float]:
        return (self["x"], self["y"], self["z"])

    @property
    def in_gray_matter(self) -> bool:
        return self.get("tissue") == "gray"

    @property
    def in_white_matter(self) -> bool:
        return self.get("tissue") == "white"

    def to_dict(self) -> Dict[str, Any]:
        return dict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Electrode":
        extra = dict(data)
        x = extra.pop("x")
        y = extra.pop("y")
        z = extra.pop("z")
        shaft = extra.pop("shaft")
        view = extra.pop("view", None)
        name = extra.pop("name", None)
        visible = extra.pop("visible", True)
        detected = extra.pop("detected", False)
        space = extra.pop("space", "unregistered")
        tissue = extra.pop("tissue", "unknown")
        slice_axial = extra.pop("slice_axial", None)
        slice_coronal = extra.pop("slice_coronal", None)
        slice_sagittal = extra.pop("slice_sagittal", None)
        extra.pop("3d_pos", None)
        return cls(
            x=x,
            y=y,
            z=z,
            shaft=shaft,
            view=view,
            name=name,
            visible=visible,
            detected=detected,
            space=space,
            tissue=tissue,
            slice_axial=slice_axial,
            slice_coronal=slice_coronal,
            slice_sagittal=slice_sagittal,
            **extra,
        )
