import io

from typing import ClassVar, Callable
from dataclasses import dataclass, fields
from pathlib import Path

from ..data_model.dxgi_format import DXGIFormat
from ..data_model.byte_buffer import BufferLayout, BufferSemantic, AbstractSemantic, Semantic, InputSlotClass
from .types import Topology
from .filename_descriptors import MigotoIndexBufferDescriptor, MigotoVertexBufferDescriptor


@dataclass
class MigotoFormat:
    # Common
    byte_offset: int = 0
    topology: Topology | None = None
    # IB
    format: DXGIFormat | None = None
    first_index: int = 0
    index_count: int = 0
    # VB
    stride: int = 0
    first_vertex: int = 0
    vertex_count: int = 0
    first_instance: int = 0
    instance_count: int = 0
    layout_hash: str | None = None
    # Semantics
    ib_layout: BufferLayout | None = None
    vb_layout: BufferLayout | None = None

    _HEADER_CONVERTERS: ClassVar[dict[str, Callable]] = {
        'topology': lambda value: Topology(value),
        'format': lambda value: DXGIFormat(value.replace('DXGI_FORMAT_', '')),
    }

    _ELEMENT_CONVERTERS: ClassVar[dict[str, Callable]] = {
        'SemanticName': lambda value: Semantic(value),
        'Format': lambda value: DXGIFormat(value.replace('DXGI_FORMAT_', '')),
        'InputSlotClass': lambda value: InputSlotClass(value),
    }

    def __post_init__(self):
        self.verify_migoto_format()

    def get_byte_offset(self, from_ib_layout: bool | None = None) -> int:

        if from_ib_layout is None:

            if self.vb_layout and self.ib_layout:
                raise ValueError(f'Byte offset calculation for MigotoFormat with both IB and VB layout requires `from_ib_layout` specified!')

            if self.ib_layout:
                from_ib_layout = True
            else:
                from_ib_layout = False

        if from_ib_layout:
            return self.byte_offset + self.format.byte_width * self.first_index
        else:
            return self.byte_offset + (self.stride or self.vb_layout.calculate_stride()) * self.first_vertex

    def get_byte_size(self, from_ib_layout: bool | None = None) -> int:

        if from_ib_layout is None:

            if self.vb_layout and self.ib_layout:
                raise ValueError(f'Byte size calculation for MigotoFormat with both IB and VB layout requires `from_ib_layout` specified!')

            if self.ib_layout:
                from_ib_layout = True
            else:
                from_ib_layout = False

        if from_ib_layout:
            return self.format.byte_width * self.index_count
        else:
            return (self.stride or self.vb_layout.calculate_stride()) * self.vertex_count

    @classmethod
    def from_migoto_descriptor(
            cls,
            migoto_filename: MigotoIndexBufferDescriptor | MigotoVertexBufferDescriptor
    ) -> 'MigotoFormat':
        fmt = cls()

        # Common
        fmt.byte_offset = migoto_filename.byte_offset or 0
        fmt.topology = migoto_filename.topology
        # IB
        if isinstance(migoto_filename, MigotoIndexBufferDescriptor):
            if migoto_filename.topology:
                fmt.format = DXGIFormat(migoto_filename.data_format.name.replace('DXGI_FORMAT_', ''))
            fmt.first_index = migoto_filename.first_index or 0
            fmt.index_count = migoto_filename.index_count or 0
            fmt.ib_layout = cls.make_ib_layout(format=fmt.format, topology=fmt.topology)
        # VB
        elif isinstance(migoto_filename, MigotoVertexBufferDescriptor):
            fmt.stride = migoto_filename.stride or 0
            fmt.first_vertex = migoto_filename.first_vertex or 0
            fmt.vertex_count = migoto_filename.vertex_count or 0
            fmt.first_instance = migoto_filename.first_instance or 0
            fmt.instance_count = migoto_filename.instance_count or 0
            fmt.layout_hash = migoto_filename.layout_hash

        return fmt

    @classmethod
    def from_paths(
            cls,
            fmt_path: Path | None = None,
            vb_path: Path | None = None,
            ib_path: Path | None = None,
    ) -> 'MigotoFormat':

        # Try to auto-detect fmt path from VB path
        if fmt_path is None and vb_path and vb_path.is_file():
            fmt_path = vb_path.with_suffix('.fmt')

        # Try to auto-detect fmt path from IB path
        if fmt_path is None and ib_path and ib_path.is_file():
            fmt_path = ib_path.with_suffix('.fmt')

        # Raise exceptions if fmt file resolution failed
        if fmt_path is None:
            raise ValueError(f'Failed to resolve format file for VB `{vb_path}` and IB `{ib_path}`')
        if not fmt_path.is_file():
            raise FileNotFoundError(f'Format file does not exist: {fmt_path}')

        # Read migoto format from fmt file
        with open(fmt_path, 'r') as fmt_file:
            fmt = MigotoFormat.from_fmt_file(fmt_file)

        return fmt

    @classmethod
    def from_dict(cls, migoto_data: dict) -> 'MigotoFormat':
        # Tokenize header data
        tokenized_headers_data = cls.tokenize_data(migoto_data, cls._HEADER_CONVERTERS)

        # Initialize instance with header data
        fmt = cls(**{
            f.name: tokenized_headers_data[f.name]
            for f in fields(cls)
            if f.name in tokenized_headers_data
        })

        # Fill IB layout ("format" field always carries INDEX0 one)
        if fmt.format is not None:
            fmt.ib_layout = cls.make_ib_layout(format=fmt.format, topology=fmt.topology)

        # Get layout data and exit early if not found
        elements_data = migoto_data.get('elements', None)
        if elements_data is None:
            return fmt

        # Tokenize elements data
        tokenized_elements_data = {}
        for element_id, element_data in migoto_data['elements'].items():
            tokenized_elements_data[element_id] = cls.tokenize_data(element_data, cls._ELEMENT_CONVERTERS)

        # Fill instance with elements data

        layout = BufferLayout(semantics=[], auto_stride=False, auto_offsets=False)

        for element in tokenized_elements_data.values():
            buffer_semantic = BufferSemantic(
                abstract=AbstractSemantic(
                    semantic=element['SemanticName'],
                    semantic_index=element['SemanticIndex'],
                ),
                format=element['Format'],
                input_slot=element['InputSlot'],
                offset=element['AlignedByteOffset'],
                input_slot_class=element['InputSlotClass'],
                instance_data_step_rate=element['InstanceDataStepRate'],
            )

            layout.add_element(buffer_semantic)

        fmt.vb_layout = layout

        return fmt

    @staticmethod
    def make_ib_layout(format: DXGIFormat, topology: Topology | None) -> BufferLayout:
        index_semantic = BufferSemantic(AbstractSemantic(Semantic.Index, 0), format=format)
        # Auto-adjust semantic stride for common topologies
        # 3dmigoto writes only R component here (i.e. R16_UINT with for R16G16B16_UINT)
        if topology == Topology.TriangleList:
            index_semantic.stride = 3 * index_semantic.format.value_byte_width
        elif topology == Topology.LineList:
            index_semantic.stride = 2 * index_semantic.format.value_byte_width
        return BufferLayout(semantics=[index_semantic], auto_stride=True, auto_offsets=False)

    @classmethod
    def parse_fmt_text(cls, lines: str) -> dict:
        migoto_data = {}
        elements_data = {}
        current_element = None

        for line in lines.splitlines():
            line: str = line.lstrip()

            if not line:
                continue

            if ':' not in line:
                # raise ValueError(f'separator `:` not found in `{line}`')
                break

            if line.startswith('element['):
                end = line.find(']')
                if end == -1:
                    raise ValueError(f'element line is corrupted (missing `]`): `{line}`')
                start = len('element[')
                element_id = int(line[start:end].strip())
                current_element = {}
                elements_data[element_id] = current_element
                continue

            k, v = map(str.strip, line.split(':', 1))

            if current_element is not None:
                current_element[k] = v
            else:
                migoto_data[k.replace(' ', '_')] = v

        migoto_data['elements'] = elements_data

        return migoto_data

    @classmethod
    def from_fmt_text(cls, text: str) -> 'MigotoFormat':
        migoto_data = cls.parse_fmt_text(text)
        return cls.from_dict(migoto_data)

    @classmethod
    def from_fmt_file(cls, file_data: io.IOBase) -> 'MigotoFormat':
        migoto_data = cls.parse_fmt_text(file_data.read())
        return cls.from_dict(migoto_data)

    @classmethod
    def extract_txt_file_fmt_text(cls, file_data: io.IOBase) -> str:
        lines = ''
        for line in file_data:
            if not line.strip():
                continue
            if ':' not in line:
                break
            if line.startswith(('vertex-data', 'instance-data')):
                break
            lines += line
        return lines

    @classmethod
    def from_txt_file(cls, file_data: io.IOBase) -> 'MigotoFormat':
        fmt_text = cls.extract_txt_file_fmt_text(file_data)
        migoto_data = cls.parse_fmt_text(fmt_text)
        return cls.from_dict(migoto_data)

    @staticmethod
    def tokenize_data(element_data: dict[str, str], converters: dict[str, Callable]) -> dict:
        tokenized_data = {}
        for k, v in element_data.items():
            converter = converters.get(k, None)
            if converter:
                tokenized_data[k] = converter(v)
            elif isinstance(v, str):
                tokenized_data[k] = int(v)
        return tokenized_data

    def verify_migoto_format(self):
        pass

    @classmethod
    def from_layouts(
            cls,
            ib_layout: BufferLayout | None = None,
            vb_layout: BufferLayout | None = None,
            topology: Topology | None = None,
    ) -> 'MigotoFormat':
        fmt = cls(
            topology=topology,
            ib_layout=ib_layout,
            vb_layout=vb_layout,
        )
        if ib_layout:
            fmt.format = ib_layout.get_element(Semantic.Index).format
        if vb_layout:
            fmt.stride = vb_layout.stride
        return fmt

    def to_blender_addon_string(self):
        """

        """

        if self.format:
            ib_format = self.format
        elif self.ib_layout:
            ib_format = self.ib_layout.get_element(Semantic.Index).format
        else:
            raise ValueError(f"IB format not defined!")

        if ib_format.value_byte_width == 4:
            ib_export_format = DXGIFormat.R32_UINT
        else:
            ib_export_format = DXGIFormat.R16_UINT

        fmt = ''
        fmt += f'stride: {self.vb_layout.stride}\n'
        fmt += f'topology: {self.topology}\n'
        fmt += f'format: {ib_export_format.get_format()}\n'
        fmt += self.vb_layout.to_string()

        return fmt


class MigotoFmt:
    def __init__(self, fmt_file: io.IOBase):
        """
        DEPRECATED!!! Use MigotoFormat instead!
        This class approach is based on assumption that all elements declare single continuos buffer
        While it's valid for native toolkit FMTs, it falls short when it comes to TXT headers parsing
        """
        fmt = MigotoFormat.from_fmt_file(fmt_file)

        # Relay layouts
        self.ib_layout = fmt.ib_layout
        self.vb_layout = fmt.vb_layout

        # Calculate per-element stride
        vb_stride = 0
        vb_byte_offset = 0
        for i, element in enumerate(fmt.vb_layout.semantics):
            next_element_id = i + 1
            if len(fmt.vb_layout.semantics) > next_element_id:
                next_offset = fmt.vb_layout.semantics[next_element_id].offset
            else:
                next_offset = vb_stride
            element.stride = next_offset - vb_byte_offset
            vb_byte_offset = next_offset

        if vb_stride != self.vb_layout.stride:
            raise ValueError(f'vb buffer layout format stride mismatch: {vb_stride} != {self.vb_layout.stride}')
