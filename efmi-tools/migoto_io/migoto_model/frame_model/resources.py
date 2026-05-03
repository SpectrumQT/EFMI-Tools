from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path

from ...data_model.byte_buffer import BufferLayout, BufferSemantic, AbstractSemantic, Semantic, NumpyBuffer

from ..helpers import raise_with_args
from ..types import SlotType, ResourceSlot, Topology, DXGI_FORMAT
from ..filename_descriptors import MigotoResourceDescriptor, MigotoBufferDescriptor, MigotoIndexBufferDescriptor, MigotoVertexBufferDescriptor
from ..migoto_format import MigotoFormat


@dataclass
class Resource:
    hash: str
    pointer: str
    usage: dict[int, list[ResourceSlot]] | None = None
    # usage: dict[int, list[ResourceSlot]] | None = field(default_factory=lambda: defaultdict(list))

    bin_path: Path | None = None
    bin_path_deduped: Path | None = None
    txt_path: Path | None = None
    txt_path_deduped: Path | None = None

    usage_descriptor: MigotoBufferDescriptor | None = None
    data_descriptor: MigotoResourceDescriptor | None = None

    buffer: NumpyBuffer | None = None
    parent: "Resource | None" = None

    def build_numpy_buffer(
        self,
        migoto_format: MigotoFormat | None = None,
        from_ib_format: bool = False,
        from_txt: bool = False,
        txt_remapped_semantics: dict[AbstractSemantic, AbstractSemantic] | None = None,
    ) -> NumpyBuffer:

        layout = migoto_format.ib_layout if from_ib_format else migoto_format.vb_layout

        if not from_txt:
            buffer = NumpyBuffer(layout)

            with open(self.bin_path, 'rb') as f:
                # Start read from specified offset
                byte_offset = migoto_format.get_byte_offset(from_ib_layout=from_ib_format)
                if byte_offset != 0:
                    f.seek(byte_offset)
                # Calculate read size from layout stride and elements count
                if not self.usage_descriptor.resource_hash.startswith("UNKNOWN_"):
                    byte_size = migoto_format.get_byte_size(from_ib_layout=from_ib_format)
                else:
                    byte_size = 0
                # Import bytes
                buffer.import_raw_data(f.read(byte_size or -1))

        else:
            if from_ib_format:
                element_count = int(migoto_format.index_count / 3)
                if migoto_format.topology != Topology.TriangleList:
                    raise ValueError(f'Expected {Topology.TriangleList} IB topology, got {migoto_format.topology}!')
            else:
                element_count = migoto_format.vertex_count

            buffer = NumpyBuffer(layout, size=element_count)

            with open(self.txt_path, 'r') as f:
                buffer.import_txt_data(f.read(), remapped_semantics=txt_remapped_semantics, is_ib=from_ib_format)

        self.buffer = buffer

        return buffer


@dataclass
class ConstantBuffer(Resource):
    first_constant: int = 0
    num_constants: int = 0

    def build_numpy_buffer(
        self,
        migoto_format: MigotoFormat | None = None,
        from_ib_format: bool = False,
        from_txt: bool = False,
        txt_remapped_semantics: dict[AbstractSemantic, AbstractSemantic] | None = None,
    ) -> NumpyBuffer:
        try:
            buffer = super().build_numpy_buffer(
                migoto_format=migoto_format,
                from_ib_format=from_ib_format,
                from_txt=from_txt,
                txt_remapped_semantics=txt_remapped_semantics,
            )

        except Exception as e:
            raise_with_args(f'Failed to create ConstantBuffer: {str(e)}!', e)

        return buffer


@dataclass
class MigotoBuffer(Resource):
    data_descriptor: MigotoIndexBufferDescriptor | MigotoVertexBufferDescriptor | None = None
    migoto_format: MigotoFormat | None = None

    views: dict[tuple[int, int], "IndexBuffer | VertexBuffer"] = field(default_factory=dict)

    def load_format(self, from_file: bool = False, reload: bool = False, txt_path: Path | None = None):
        # TODO: Add MigotoFormat cache with filename as key (format declared by call may be different for the same resource)
        # if self.migoto_format and not reload:
        #     return
        if from_file:
            with open(txt_path or self.txt_path, 'r') as f:
                self.migoto_format = MigotoFormat.from_txt_file(f)
        else:
            self.migoto_format = MigotoFormat.from_migoto_descriptor(self.data_descriptor)

    def create_view(self, view_key) -> "IndexBuffer | VertexBuffer":
        if isinstance(self.data_descriptor, MigotoIndexBufferDescriptor):
            view_type = IndexBuffer
        elif isinstance(self.data_descriptor, MigotoVertexBufferDescriptor):
            view_type = VertexBuffer
        else:
            raise ValueError

        view = view_type(
            pointer=f"{self.pointer}:{view_key[0]}:{view_key[1]}",
            parent=self,
            hash=self.usage_descriptor.resource_hash,
            migoto_format=self.migoto_format,
            usage_descriptor=self.usage_descriptor,
            data_descriptor=self.data_descriptor,
            bin_path=self.bin_path,
            bin_path_deduped=self.bin_path_deduped,
            txt_path=self.txt_path,
            txt_path_deduped=self.txt_path_deduped,
        )

        self.views[view_key] = view

        return view

    def get_view(self) -> "IndexBuffer | VertexBuffer":

        byte_offset = self.migoto_format.get_byte_offset()
        byte_size = self.migoto_format.get_byte_size()

        view_key = (byte_offset, byte_size)

        view = self.views.get(view_key, None)

        if view is None:
            view = self.create_view(view_key)

        return view


@dataclass
class VertexBuffer(MigotoBuffer):
    data_descriptor: MigotoVertexBufferDescriptor | None = None

    def build_numpy_buffer(
        self,
        migoto_format: MigotoFormat | None = None,
        from_ib_format: bool = False,
        from_txt: bool = False,
        txt_remapped_semantics: dict[AbstractSemantic, AbstractSemantic] | None = None,
    ) -> NumpyBuffer:
        try:
            if migoto_format is None:
                self.load_format()
                migoto_format = self.migoto_format

            if migoto_format.vb_layout is None:
                self.load_format(from_file=True, reload=True)

            buffer = super().build_numpy_buffer(
                migoto_format=migoto_format,
                from_ib_format=False,
                from_txt=from_txt,
                txt_remapped_semantics=txt_remapped_semantics,
            )

            if migoto_format.vertex_count == 0 or self.usage_descriptor.resource_hash.startswith("UNKNOWN_"):
                migoto_format.vertex_count = len(buffer.data)

            num_elements = migoto_format.vertex_count

            # Format may expect a slice of a .buf file, while .txt files are always already sliced
            # So if format defines non-zero first_index and/or index_count, we must slice the .buf data accordingly
            if not from_txt and len(buffer.data) != num_elements:
                offset = migoto_format.first_vertex
                buffer.data = buffer.data[offset:offset + num_elements]

            if len(buffer.data) != num_elements:
                raise ValueError(f'vertex count {len(buffer.data)} differs from expected {num_elements}')

        except Exception as e:
            raise_with_args(f'Failed to create VertexBuffer: {str(e)}!', e)

        return buffer

    def __repr__(self):
        repr = f"hash={self.hash}, pointer={self.pointer}"

        semantics = []
        if self.migoto_format and self.data_descriptor:
            if self.migoto_format.vb_layout is not None:
                for buffer_semantic in self.migoto_format.vb_layout.semantics:
                    if buffer_semantic.input_slot == self.data_descriptor.slot_id:
                        semantics.append(str(buffer_semantic))
            if semantics:
                layout = ", ".join(semantics)
                repr += f", layout=[{layout}]"

        return repr


@dataclass
class IndexBuffer(MigotoBuffer):
    byte_offset: int = 0
    first_index: int = 0
    index_count: int = 0
    topology: Topology = Topology.Undefined
    format: DXGI_FORMAT = DXGI_FORMAT.DXGI_FORMAT_UNKNOWN

    data_descriptor: MigotoIndexBufferDescriptor | None = None

    def build_numpy_buffer(
        self,
        migoto_format: MigotoFormat | None = None,
        from_ib_format: bool = False,
        from_txt: bool = False,
        txt_remapped_semantics: dict[AbstractSemantic, AbstractSemantic] | None = None,
    ) -> NumpyBuffer:
        try:
            if migoto_format is None:
                self.load_format()
                migoto_format = self.migoto_format

            if migoto_format.ib_layout is None:
                self.load_format(from_file=True, reload=True)

            buffer = super().build_numpy_buffer(
                migoto_format=migoto_format,
                from_ib_format=True,
                from_txt=from_txt,
                txt_remapped_semantics=txt_remapped_semantics,
            )

            data = buffer.get_field(Semantic.Index)
            second_dim_size = data.shape[1] if data.ndim == 2 else 1

            if migoto_format.index_count == 0 or self.usage_descriptor.resource_hash.startswith("UNKNOWN_"):
                migoto_format.index_count = len(data) * second_dim_size

            # Ensure sanity of expected data
            if migoto_format.first_index % second_dim_size != 0:
                raise ValueError(f'first index {migoto_format.first_index} is not dividable by {second_dim_size}')
            if migoto_format.index_count % second_dim_size != 0:
                raise ValueError(f'index count {migoto_format.index_count} is not dividable by {second_dim_size}')

            num_elements = int(migoto_format.index_count / second_dim_size)

            # Format may expect a slice of a .buf file, while .txt files are always already sliced
            # So if format defines non-zero first_index and/or index_count, we must slice the .buf data accordingly
            if not from_txt and len(buffer.data) != num_elements:
                offset = int(migoto_format.first_index / second_dim_size)
                buffer.data = buffer.data[offset:offset+num_elements]

            if len(buffer.data) != num_elements:
                raise ValueError(f'index count {len(data)} differs from expected {num_elements}')

        except Exception as e:
            raise_with_args(f'Failed to create IndexBuffer: {str(e)}!', e)

        return buffer


@dataclass
class ResourceStorage:
    _pointer_index: dict[str, Resource] = field(default_factory=dict)

    # hash -> Resource | set[Resource] (hybrid collision handling)
    _hash_index: dict[str, Resource | list[Resource]] = field(default_factory=dict)

    _resource_to_slot: dict[str, ResourceSlot] = field(default_factory=dict)

    constant_buffers: dict[ResourceSlot, ConstantBuffer] = field(default_factory=dict)
    index_buffer: dict[ResourceSlot, IndexBuffer] = field(default_factory=dict)
    vertex_buffers: dict[ResourceSlot, VertexBuffer] = field(default_factory=dict)
    textures: dict[ResourceSlot, Resource] = field(default_factory=dict)
    unordered_access_views: dict[ResourceSlot, Resource] = field(default_factory=dict)
    render_targets: dict[ResourceSlot, Resource] = field(default_factory=dict)

    def __post_init__(self):
        self._slot_type_index = {
            SlotType.Texture: self.textures,
            SlotType.UAV: self.unordered_access_views,
            SlotType.ConstantBuffer: self.constant_buffers,
            SlotType.IndexBuffer: self.index_buffer,
            SlotType.VertexBuffer: self.vertex_buffers,
            SlotType.RenderTarget: self.render_targets,
        }

    def add(self, slot: ResourceSlot | str, resource: Resource) -> None:
        current_resource = self.get_by_slot(slot)
        if current_resource is not None:
            self.remove(resource=current_resource, slot=slot)

        self._slot_type_index[slot.slot_type][slot] = resource

        self._resource_to_slot[resource.pointer] = slot

        self._pointer_index[resource.pointer] = resource

        entry = self._hash_index.get(resource.hash)

        if entry is None:
            self._hash_index[resource.hash] = resource
        elif isinstance(entry, list):
            entry.append(resource)
        else:
            self._hash_index[resource.hash] = [entry, resource]

    def remove(self, resource: Resource | ResourceSlot, slot: ResourceSlot | None = None) -> None:
        if isinstance(resource, Resource):
            if slot is None:
                slot = self._resource_to_slot.pop(resource.pointer, None)
        elif isinstance(resource, ResourceSlot):
            slot = resource
            resource = self.get_by_slot(slot)
        else:
            raise ValueError(f'removal by {type(resource)} resource type is not supported')

        if slot is not None:
            self._slot_type_index[slot.slot_type].pop(slot, None)

        self._pointer_index.pop(resource.pointer, None)

        entry = self._hash_index.get(resource.hash)

        if isinstance(entry, list):
            entry = [x for x in entry if x != resource]
            if len(entry) == 1:
                self._hash_index[resource.hash] = next(iter(entry))
            elif not entry:
                self._hash_index.pop(resource.hash, None)

        elif entry is resource:
            self._hash_index.pop(resource.hash, None)

    def get_slot_index(self, slot_type: SlotType) -> dict:
        return self._slot_type_index[slot_type]

    def get_by_slot(self, slot: ResourceSlot | str) -> Resource | None:
        if isinstance(slot, str):
            slot = ResourceSlot.from_string(slot)
        return self._slot_type_index[slot.slot_type].get(slot, None)

    def get_by_pointer(self, pointer: str) -> Resource:
        return self._pointer_index.get(pointer)

    def get_by_hash(self, hash_: str) -> list[Resource] | None:
        entry = self._hash_index.get(hash_)

        if entry is None:
            return None

        if isinstance(entry, list):
            return entry

        return [entry]

    def get_slot(self, resource: Resource):
        return self._resource_to_slot.get(resource.pointer)
