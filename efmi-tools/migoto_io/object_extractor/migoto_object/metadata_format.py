import json

from typing import get_origin, get_args
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field, asdict, fields, is_dataclass

from ...data_model.dxgi_format import DXGIFormat
from ...data_model.byte_buffer import Semantic, AbstractSemantic, BufferSemantic, BufferLayout


@dataclass
class ExtractedObjectBufferSemantic:
    name: Semantic
    index: int
    format: DXGIFormat
    stride: int = 0

    def __post_init__(self):
        if isinstance(self.name, str):
            self.name = Semantic(self.name.upper())
        if isinstance(self.format, str):
            self.format = DXGIFormat(self.format.upper())
        if self.stride == 0:
            self.stride = self.format.byte_width

    @classmethod
    def from_buffer_semantic(cls, buffer_semantic: BufferSemantic):
        return cls(
            name=buffer_semantic.abstract.enum,
            index=buffer_semantic.abstract.index,
            format=buffer_semantic.format,
            stride=buffer_semantic.stride,
        )

    def to_buffer_semantic(self) -> BufferSemantic:
        return BufferSemantic(
            abstract=AbstractSemantic(self.name, self.index),
            format=self.format,
            stride=self.stride,
        )


@dataclass
class ExtractedObjectBuffer:
    semantics: list[ExtractedObjectBufferSemantic]

    @classmethod
    def from_buffer_layout(cls, layout: BufferLayout):
        return ExtractedObjectBuffer(
            semantics=[
                ExtractedObjectBufferSemantic.from_buffer_semantic(semantic) for semantic in layout.semantics
            ]
        )

    def to_layout(self) -> BufferLayout:
        layout = BufferLayout([])
        for semantic in self.semantics:
            layout.add_element(semantic.to_buffer_semantic())
        return layout


@dataclass
class ExtractedObjectComponentLOD:
    lod_object_name: str
    ib_hash: str
    vb0_hash: str
    vertex_offset: int
    vertex_count: int
    index_offset: int
    index_count: int
    vg_map: dict[int, int]
    vb_formats: dict[str, ExtractedObjectBuffer]


@dataclass
class ExtractedObjectComponent:
    mesh_name: str
    cpu_posed: bool
    ib_hash: str
    vb0_hash: str
    vertex_offset: int
    vertex_count: int
    index_offset: int
    index_count: int
    vg_offset: int
    vg_count: int
    vg_map: dict[int, int]
    lods: list[ExtractedObjectComponentLOD]


@dataclass
class ExtractedObjectShapeKeys:
    offsets_hash: str = ''
    scale_hash: str = ''
    vertex_count: int = 0
    dispatch_y: int = 0
    checksum: int = 0


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


@dataclass
class ObjectRotation:
    x: int = 0
    y: int = 0
    z: int = 0

    def to_tuple(self) -> tuple[int, int, int]:
        return self.x, self.y, self.z


@dataclass
class ExtractedObject:
    ib_hash: str
    vb0_hash: str
    vertex_count: int
    index_count: int
    rotation: ObjectRotation
    components: list[ExtractedObjectComponent]
    shapekeys: ExtractedObjectShapeKeys
    export_format: dict[str, ExtractedObjectBuffer]

    def __post_init__(self):
        if self.rotation is None:
            self.rotation = ObjectRotation()

        for component_id, component in enumerate(self.components):
            if component.mesh_name is None:
                component.mesh_name = f"Component {component_id}"

    def as_json(self):
        return json.dumps(asdict(self), indent=4, cls=EnumEncoder)
    

def from_dict(cls, data):
    if not is_dataclass(cls):
        return data

    kwargs = {}
    for f in fields(cls):
        value = data.get(f.name)
        if value is None:
            kwargs[f.name] = None
            continue

        field_type = f.type
        origin = get_origin(field_type)

        if origin is list:
            item_type = get_args(field_type)[0]
            kwargs[f.name] = [from_dict(item_type, v) for v in value]
        elif origin is dict:
            key_type, val_type = get_args(field_type)
            kwargs[f.name] = {k: from_dict(val_type, v) for k, v in value.items()}
        else:
            kwargs[f.name] = from_dict(field_type, value)

    return cls(**kwargs)


def read_metadata(metadata_path: Path) -> ExtractedObject:
    with open(metadata_path) as f:
        return from_dict(ExtractedObject, json.load(f))
