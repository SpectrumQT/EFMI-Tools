from pathlib import Path
from dataclasses import dataclass, field

from ...migoto_model.types import SlotType, ResourceSlot
from ...migoto_model.frame_model.resources import Resource
from ...migoto_model.migoto_mesh import MigotoMesh, WeightingType

from ..raw_object.raw_object import RawComponent

from .metadata_format import ExtractedObject, ExtractedObjectComponent, ExtractedObjectShapeKeys, ExtractedObjectComponentLOD, ObjectRotation, ExtractedObjectBuffer
from .metadata_format import read_metadata


class DuplicateDataError(Exception):
    pass


@dataclass(eq=False)
class MigotoComponent:
    mesh: MigotoMesh
    textures: dict[ResourceSlot, list[Resource]]
    raw_data: RawComponent | None = None
    metadata: ExtractedObjectComponent | None = None

    def build_metadata(self, component_name: str):
        self.metadata = ExtractedObjectComponent(
            mesh_name=component_name,
            cpu_posed=self.mesh.cpu_posed,
            ib_hash=self.raw_data.shader_calls[0].resources.get_by_slot("ib").hash,
            vb0_hash=self.raw_data.shader_calls[0].resources.get_by_slot("vb0").hash,
            vertex_offset=0,
            vertex_count=self.mesh.format.vertex_count,
            index_offset=0,
            index_count=self.mesh.format.index_count,
            vg_offset=0,
            vg_count=0,
            vg_map={},
            lods=[],
        )

    def import_lod_metadata(
        self,
        lod_object_name: str,
        lod_component: "MigotoComponent",
        vg_map: dict[int, int] | None,
        allow_overwrite: bool = False,
    ):
        
        full_layout = self.mesh.vertex_buffer.layout
        lod_layout = lod_component.mesh.vertex_buffer.layout

        input_slots = full_layout.get_input_slots() | lod_layout.get_input_slots()

        vb_formats = {}
        for input_slot in input_slots:
            full_slot_layout = full_layout.get_input_slot_layout(input_slot)
            lod_slot_layout = lod_layout.get_input_slot_layout(input_slot)
            if full_slot_layout.to_string() != lod_slot_layout.to_string():
                vb_formats[f"VB{input_slot}"] = ExtractedObjectBuffer.from_buffer_layout(lod_slot_layout)

        new_lod_metadata = ExtractedObjectComponentLOD(
            lod_object_name=lod_object_name,
            ib_hash=lod_component.metadata.ib_hash,
            vb0_hash=lod_component.metadata.vb0_hash,
            vertex_offset=lod_component.metadata.vertex_offset,
            vertex_count=lod_component.metadata.vertex_count,
            index_offset=lod_component.metadata.index_offset,
            index_count=lod_component.metadata.index_count,
            vg_map=vg_map,
            vb_formats=vb_formats,
        )

        lods = []
        for lod_metadata in self.metadata.lods:
            if lod_metadata.lod_object_name == new_lod_metadata.lod_object_name:
                if not allow_overwrite:
                    raise DuplicateDataError(f"{self.metadata.mesh_name} already contains LoD imported from {lod_object_name}!")
            elif lod_metadata.vertex_count == new_lod_metadata.vertex_count:
                if not allow_overwrite:
                    raise DuplicateDataError(f"{self.metadata.mesh_name} already contains LoD with {lod_metadata.vertex_count} vertices!")
            else:
                lods.append(lod_metadata)

        lods.append(new_lod_metadata)

        lods.sort(key=lambda lod: lod.vertex_count, reverse=True)

        self.metadata.lods = lods

    def __repr__(self):
        if self.metadata is not None:
            return f"{self.metadata.mesh_name} (ib_hash={self.metadata.ib_hash}, vertex_count={self.metadata.vertex_count})"
        return super().__repr__()


@dataclass(eq=False)
class MigotoObject:
    id: str
    components: list[MigotoComponent] = field(default_factory=list)
    metadata: ExtractedObject | None = None

    def build_metadata(self):
        is_weighted = any(component.mesh.get_weighting_type() != WeightingType.NoWeights for component in self.components)
        has_vertex_offset = any(component.raw_data.vertex_offset != 0 for component in self.components)
        if has_vertex_offset or not is_weighted:
            rotation = ObjectRotation(90, 0, 0)
        else:
            rotation = ObjectRotation(0, 0, 0)

        for component_id, component in enumerate(self.components):
            component.build_metadata(f"Component {component_id}")

        self.metadata = ExtractedObject(
            ib_hash=None,
            vb0_hash=None,
            vertex_count=sum([component.mesh.format.vertex_count for component in self.components]),
            index_count=sum([component.mesh.format.index_count for component in self.components]),
            rotation=rotation,
            components=[component.metadata for component in self.components],
            shapekeys=ExtractedObjectShapeKeys(),
            export_format={},
        )

    def import_metadata(self, metadata: ExtractedObject):
        if len(self.components) != len(metadata.components):
            raise ValueError(f"Components count mismatch between mesh ({len(self.components)}) and metadata ({len(metadata.components)})!")
        for component_id, component_metadata in enumerate(metadata.components):
            self.components[component_id].metadata = component_metadata
        self.metadata = metadata

    def export_metadata(self, folder_path: Path):
        folder_path.mkdir(parents=True, exist_ok=True)
        with open(folder_path / f'Metadata.json', "w") as f:
            f.write(self.metadata.as_json())

    def export_to_files(self, folder_path: Path):
        self.export_metadata(folder_path)

        for component in self.components:
            component.mesh.export_as_migoto_raw_buffers(folder_path, component.metadata.mesh_name)

    @staticmethod
    def index_mesh_paths(folder_path: Path) -> dict[str, Path]:
        mesh_paths = {}
        for file_path in folder_path.iterdir():
            if not file_path.is_file() or file_path.suffix != ".fmt":
                continue
            mesh_paths[file_path.stem] = file_path
        return mesh_paths

    @classmethod
    def from_exported_files(cls, object_folder_path: Path, metadata_path: Path | None = None):

        mesh_paths = cls.index_mesh_paths(object_folder_path)

        if metadata_path is None:
            metadata_path = object_folder_path / "Metadata.json"

        try:
            metadata = read_metadata(object_folder_path / "Metadata.json") if metadata_path.is_file() else None
        except Exception as e:
            raise ValueError(f'Failed to load Metadata.json:\n{e}')

        if metadata is not None:
            defined_mesh_paths = {}
            for component_id, component_metadata in enumerate(metadata.components):
                mesh_path = mesh_paths.get(component_metadata.mesh_name,None)
                if mesh_path is None:
                    raise ValueError(f"Mesh `{component_metadata.mesh_name}` defined in {metadata_path.name}/Component {component_id} not found in {object_folder_path}!")
                defined_mesh_paths[component_metadata.mesh_name] = mesh_path
            mesh_paths = defined_mesh_paths

        meshes = {}
        for mesh_name, mesh_path in mesh_paths.items():
            meshes[mesh_name] = MigotoMesh.from_paths(fmt_path=mesh_path)

        migoto_object = cls(
            id=object_folder_path.name,
            components=[
                MigotoComponent(
                    mesh=mesh,
                    textures={},
                ) for mesh in meshes.values()
            ],
        )

        if metadata is not None:
            migoto_object.import_metadata(metadata)

        return migoto_object
