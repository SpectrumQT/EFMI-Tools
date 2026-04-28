import io
import copy
import textwrap
import math
import numpy
import re

from dataclasses import dataclass
from operator import attrgetter
from typing import Callable

from .dxgi_format import *


SEMANTIC_ALIASES = {
    'BLENDWEIGHT': 'BLENDWEIGHTS',
    'BLENDWEIGHTS': 'BLENDWEIGHT',
}

class Semantic(Enum):
    VertexId = 'VERTEXID'
    Index = 'INDEX'
    Tangent = 'TANGENT'
    BitangentSign = 'BITANGENTSIGN'
    Normal = 'NORMAL'
    TexCoord = 'TEXCOORD'
    Color = 'COLOR'
    Position = 'POSITION'
    Blendindices = 'BLENDINDICES'
    Blendweight = 'BLENDWEIGHT'
    Blendweights = 'BLENDWEIGHTS'
    ShapeKey = 'SHAPEKEY'
    RawData = 'RAWDATA'
    EncodedData = 'ENCODEDDATA'
    Attribute = 'ATTRIBUTE'
    Unknown = 'UNKNOWN'

    def __str__(self):
        return f'{self.value}'

    def __repr__(self):
        return f'{self.value}'

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other or SEMANTIC_ALIASES.get(self.value) == other
        if isinstance(other, Semantic):
            return self.value == other.value or SEMANTIC_ALIASES.get(self.value) == other.value
        return super().__eq__(other)

    def __hash__(self):
        return hash(self.value)


class InputSlotClass(Enum):
    PerVertex = 'per-vertex'
    PerInstance = 'per-instance'

    def __str__(self):
        return f'{self.value}'

    def __repr__(self):
        return f'{self.value}'


@dataclass
class AbstractSemantic:
    enum: Semantic
    index: int = 0

    def __init__(self, semantic, semantic_index=0):
        self.enum = semantic
        self.index = semantic_index

    def __hash__(self):
        return hash((self.enum, self.index))

    def __str__(self):
        return f'{self.enum}_{self.index}'

    def __repr__(self):
        return f'{self.enum}_{self.index}'
    
    def get_name(self):
        name = self.enum.value
        if self.index > 0:
            name += str(self.index)
        if self.enum == Semantic.TexCoord:
            name += '.xy'
        return name


@dataclass
class BufferSemantic:
    abstract: AbstractSemantic
    format: DXGIFormat
    stride: int = 0
    offset: int = 0
    name: str | None = None
    input_slot: int = 0
    input_slot_class: InputSlotClass = InputSlotClass.PerVertex
    instance_data_step_rate: int = 0
    import_format: DXGIFormat | None = None
    extract_format: DXGIFormat | None = None

    def __post_init__(self):
        # Calculate byte stride
        if self.stride == 0:
            self.stride = self.format.byte_width
        
    def __hash__(self):
        return hash((self.abstract, self.input_slot, self.format.format, self.stride, self.offset))

    def __repr__(self):
        return f'{self.abstract} ({self.format.format} input={self.input_slot} stride={self.stride} offset={self.offset})'

    def to_string(self, indent=2):
        return textwrap.indent(textwrap.dedent(f'''
            SemanticName: {self.abstract.enum}
            SemanticIndex: {self.abstract.index}
            Format: {self.format.format}
            InputSlot: {self.input_slot}
            AlignedByteOffset: {self.offset}
            InputSlotClass: {self.input_slot_class}
            InstanceDataStepRate: 0
        ''').lstrip(), ' ' * indent)

    def get_format(self) -> str:
        return self.format.get_format()

    def get_name(self) -> str:
        return self.name if self.name else self.abstract.get_name()

    def get_num_values(self) -> int:
        return self.format.get_num_values(self.stride)

    def get_numpy_type(self) -> Union[int, Tuple[Union[numpy.integer, numpy.floating], int]]:
        return self.format.get_numpy_type(self.stride)
    

@dataclass
class BufferLayout:
    semantics: list[BufferSemantic]
    stride: int = 0
    force_stride: bool = False
    auto_stride: bool = True
    auto_offsets: bool = True

    def __post_init__(self):
        # Autofill byte Stride and Offsets
        if self.auto_stride and self.stride == 0:
            self.fill_stride()
            if self.auto_offsets:
                self.fill_offsets()
        # Autofill Semantic Index
        groups = {}
        for semantic in self.semantics:
            if semantic not in groups:
                groups[semantic] = 0
                continue
            if semantic.abstract.index == 0:
                groups[semantic] += 1
                semantic.abstract.index = groups[semantic]

    def fill_stride(self):
        self.stride = self.calculate_stride()

    def calculate_stride(self):
        stride = 0
        for element in self.semantics:
            stride += element.stride
        return stride

    def fill_offsets(self):
        offset = 0
        for element in self.semantics:
            element.offset = offset
            offset += element.stride

    def get_element(self, element: Union[AbstractSemantic, Semantic, int]) -> BufferSemantic | None:
        
        if isinstance(element, str):
            for layout_element in self.semantics:
                if element == layout_element.get_name():
                    return layout_element
            # raise ValueError(f'Layout element with name {element} is not found!')
            return None

        if isinstance(element, int):
            if element >= len(self.semantics):
                # raise ValueError(f'Layout element with id {element} is out of 0-{len(self.semantics)} bounds!')
                return None
            return self.semantics[element]
        
        if isinstance(element, Semantic):
            element = AbstractSemantic(element)

        if not isinstance(element, AbstractSemantic):
            raise ValueError(f'Layout element search by type {type(element)} of value {element} is not supported!')
        
        for layout_element in self.semantics:
            if element == layout_element.abstract:
                return layout_element
            
        return None

    def set_element(self, abstract: AbstractSemantic, semantic: BufferSemantic):
        for i, element in enumerate(self.semantics):
            if abstract == element.abstract:
                self.stride -= element.stride
                self.semantics[i] = semantic
                self.stride += semantic.stride
                if self.auto_stride:
                    self.fill_stride()
                    if self.auto_offsets:
                        self.fill_offsets()
                return
        raise ValueError(f'Buffer semantic {abstract} not found in layout!')

    def add_element(self, semantic: BufferSemantic):
        if self.get_element(semantic.abstract) is not None:
            return
        semantic = copy.deepcopy(semantic)
        if self.auto_offsets:
            semantic.offset = self.stride
        if self.auto_stride:
            self.stride += semantic.stride
        self.semantics.append(semantic)

    def merge(self, layout: 'BufferLayout'):
        for semantic in layout.semantics:
            if not self.get_element(semantic.abstract):
                self.add_element(semantic)

    def to_string(self):
        ret = ''
        for i, semantic in enumerate(self.semantics):
            ret += 'element[%i]:\n' % i
            ret += semantic.to_string()
        return ret

    def get_numpy_type(self):
        dtype = numpy.dtype([])
        for semantic in self.semantics:
            dtype = numpy.dtype(dtype.descr + [(semantic.abstract.get_name(), (semantic.get_numpy_type()))])
        return dtype
    
    def get_max_input_slot(self) -> int:
        return max([semantic.input_slot for semantic in self.semantics])
    
    def get_elements_in_slot(self, input_slot: int) -> list[BufferSemantic]:
        return [semantic for semantic in self.semantics if semantic.input_slot == input_slot]
    
    def sort(self):
        self.semantics.sort(key=attrgetter('input_slot', 'offset'))

    def remap_semantics(self, semantic_map: dict[BufferSemantic, BufferSemantic]) -> list[tuple[BufferSemantic, BufferSemantic]]:
        remapped_semantics = []
        for semantic_id, semantic in enumerate(self.semantics):
            remapped_semantic = None
            for map_from_semantic, map_to_semantic in semantic_map.items():
                if map_from_semantic.input_slot and map_from_semantic.input_slot != semantic.input_slot:
                    continue
                if map_from_semantic.offset and map_from_semantic.offset != semantic.offset:
                    continue
                if map_from_semantic.stride != semantic.stride:
                    continue
                if map_from_semantic.format.format != semantic.format.format:
                    continue
                if map_from_semantic.abstract == semantic.abstract:
                    remapped_semantic = map_to_semantic
                    remapped_semantics.append((map_from_semantic, map_to_semantic))
                    break
            if remapped_semantic is None:
                continue
            if remapped_semantic.stride != semantic.stride:
                raise ValueError(f"Remapped semantic {remapped_semantic} stride {remapped_semantic.stride} differs from {semantic.stride} of {semantic}")

            semantic.abstract = remapped_semantic.abstract
            semantic.format = remapped_semantic.format
            semantic.input_slot = remapped_semantic.input_slot
            if remapped_semantic.offset:
                semantic.offset = remapped_semantic.offset

            # self.semantics[semantic_id] = remapped_semantic

        return remapped_semantics

    def remove_data_views(self):
        filtered_semantics = []
        for input_slot in range(self.get_max_input_slot()+1):
            semantics = self.get_elements_in_slot(input_slot)
            slot_semantics = {}
            for semantic in semantics:
                slot_semantic = slot_semantics.get(semantic.offset, None)
                if slot_semantic is not None:
                    continue
                slot_semantics[semantic.offset] = semantic
            filtered_semantics += list(slot_semantics.values())
        self.semantics = filtered_semantics

    def dedupe_semantics(self):
        filtered_semantics: list[BufferSemantic] = []
        for semantic in self.semantics:
            found = False
            for filtered_semantic in filtered_semantics:
                if semantic.input_slot != filtered_semantic.input_slot:
                    continue
                if semantic.offset != filtered_semantic.offset:
                    continue
                if semantic.stride != filtered_semantic.stride:
                    continue
                if semantic.format.format != filtered_semantic.format.format:
                    continue
                if semantic.abstract == filtered_semantic.abstract:
                    found = True
                    break
            if not found:
                filtered_semantics.append(semantic)
        self.semantics = filtered_semantics

    def fill_missing_semantics(self) -> list[BufferSemantic] | None:
        semantics_stride = self.calculate_stride()
        if self.stride <= semantics_stride:
            return None

        # Ensure layout describes single input slot.
        input_slots = set([semantic.input_slot for semantic in self.semantics])
        if len(input_slots) > 1:
            raise ValueError(f"Cannot fill missing semantics for layout with more than 1 input slot!")
        input_slot = next(iter(input_slots))

        # Sort by byte offset to ensure proper gap detection.
        self.semantics.sort(key=lambda semantic: semantic.offset)

        filled_semantics = []
        unknown_semantics = []
        current_offset = 0
        unknown_index = 0

        for i in range(len(self.semantics) + 1):
            semantic = self.semantics[i] if i < len(self.semantics) else None
            next_offset = semantic.offset if semantic else self.stride

            if next_offset > current_offset:
                unknown_semantic = BufferSemantic(
                    AbstractSemantic(Semantic.Unknown, unknown_index),
                    format=DXGIFormat.R8_UINT,
                )
                unknown_semantic.offset = current_offset
                unknown_semantic.stride = next_offset - current_offset
                unknown_semantic.input_slot = input_slot

                filled_semantics.append(unknown_semantic)
                unknown_semantics.append(unknown_semantic)
                unknown_index += 1

            if semantic:
                filled_semantics.append(semantic)
                current_offset = semantic.offset + semantic.stride

        self.semantics = filled_semantics

        return unknown_semantics


class NumpyBuffer:
    layout: BufferLayout
    data: numpy.ndarray

    def __init__(self, layout: BufferLayout, data: numpy.ndarray | None = None, size = 0):
        self.set_layout(layout)
        self.set_data(data, size)

    def set_layout(self, layout: BufferLayout):
        self.layout = layout

    def set_data(self, data: numpy.ndarray | None, size = 0):
        if data is not None:
            self.data = data
        elif size > 0:
            self.data = numpy.zeros(size, dtype=self.layout.get_numpy_type())

    def set_field(self, field: Union[AbstractSemantic, Semantic, int, str], data: numpy.ndarray | None):
        # Ensure that NumpyBuffer layout contains target semantic
        semantic = self.layout.get_element(field)
        if semantic is None:
            raise ValueError(f'Semantic {field} not found in the layout of NumpyBuffer!')
        # Ensure that NumpyBuffer data is initialized with target field 
        target_field = self.get_field(semantic.get_name())
        if target_field is None:
            raise ValueError(f'Field {field} not found in the data of NumpyBuffer!')
        try:
            # Automatically convert 2-dim array of shape "N, 1" to 1-dim array of shape "N"
            if target_field.ndim == 1:
                if data.ndim == 2 and data.shape[1] == 1:
                    data = data.squeeze(axis=1)
            # Set the named field data
            self.data[semantic.get_name()] = data
        except Exception as e:
            raise ValueError(f'Failed to set field {field} data for semantic {semantic}: {str(e)}') from e

    def get_data(self, indices: numpy.ndarray | None = None) -> numpy.ndarray:
        if indices is None:
            return self.data
        else:
            return self.data[indices]

    def get_field(self, field: Union[AbstractSemantic, Semantic, int, str]) -> numpy.ndarray | None:
        semantic = self.layout.get_element(field)
        if semantic is None:
            # raise ValueError(f'Semantic {field} not found in the layout of NumpyBuffer!')
            return None
        return self.data[semantic.get_name()]

    def remove_duplicates(self, keep_order = True):
        if keep_order:
            _, unique_index = numpy.unique(self.data, return_index=True)
            self.data = self.data[numpy.sort(unique_index)]
        else:
            self.data = numpy.unique(self.data)

    def import_semantic_data(self,
                             data: numpy.ndarray, 
                             semantic: Union[BufferSemantic, int], 
                             semantic_converters: list[Callable] | None = None,
                             format_converters: list[Callable] | None = None):
        
        if isinstance(semantic, int):
            semantic = self.layout.semantics[semantic]
        current_semantic = self.layout.get_element(semantic.abstract)
        if current_semantic is None:
            raise ValueError(f'NumpyBuffer is missing {semantic.abstract} semantic data!')
        if semantic_converters is not None:
            for data_converter in semantic_converters:
                data = data_converter(data)
        if format_converters is not None:
            for data_converter in format_converters:
                data = data_converter(data)
        if data.dtype != current_semantic.format.numpy_base_type:
            data = current_semantic.format.type_encoder(data)
        try:
            self.set_field(current_semantic.get_name(), data)
        except Exception as e:
            raise ValueError(f'Failed to import semantic {semantic} to buffer layout {self.layout}: {str(e)}') from e
        
    def import_data(self,
                    data: 'NumpyBuffer',
                    semantic_converters: dict[AbstractSemantic, list[Callable]],
                    format_converters: dict[AbstractSemantic, list[Callable]]):
        
        for buffer_semantic in self.layout.semantics:

            data_semantic = data.layout.get_element(buffer_semantic.abstract)

            if data_semantic is None:
                continue
            field_data = data.get_field(buffer_semantic.get_name())

            self.import_semantic_data(
                field_data, 
                data_semantic,
                semantic_converters.get(buffer_semantic.abstract, []),
                format_converters.get(buffer_semantic.abstract, []))
            
    def import_raw_data(self, data: numpy.ndarray | bytes):
        self.data = numpy.frombuffer(data, dtype=self.layout.get_numpy_type())

    def import_txt_data(
            self,
            data: str,
            remapped_semantics,
            is_ib: bool = False,
            ib_points_per_face: int = 3,
    ):
        remapped_semantics = remapped_semantics or {}

        # Strict matching
        # for i, semantic in enumerate(self.layout.semantics):
        #     semantic_name = remapped_semantics.get(semantic.abstract, semantic).get_name()
        #     semantic_name = semantic_name.split('.')[0]
        #     # Each number in its own capture group
        #     groups = ",".join([f"({float_pattern})" for _ in range(semantic.get_num_values())])
        #     groups = groups.replace(",", r",\s*")
        #     newline = r"\s*\n" if i < len(self.layout.semantics) - 1 else ""
        #     # Match optional prefix "vb0[...]"
        #     line_pattern = rf"vb0\[\d+\]\+\d+\s+{semantic_name}:\s*{groups}{newline}"
        #     pattern_lines.append(line_pattern)

        if is_ib:
            # Use special regex pattern for IB
            dtype = numpy.uint32
            value_pattern = r"\d+"
            groups = " ".join([f"({value_pattern})" for _ in range(ib_points_per_face)])
            full_pattern = rf"{groups}\s*\n?"

        else:
            # Build regex pattern dynamically
            dtype = numpy.float32
            value_pattern = r"[+-]?(?:\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|nan)"
            pattern_lines = []

            for semantic in self.layout.semantics:
                semantic_name = remapped_semantics.get(semantic.abstract, semantic).get_name()
                semantic_name = semantic_name.split('.')[0]
                groups = ",".join([f"({value_pattern})" for _ in range(semantic.get_num_values())])
                groups = groups.replace(",", r",\s*")
                line_pattern = rf"vb\d+\[\d+\]\+\d+\s+{semantic_name}:\s*{groups}\s*\n?"
                pattern_lines.append(line_pattern)

            # Join all lines
            full_pattern = "".join(pattern_lines)

        # Compile regex
        pattern = re.compile(full_pattern)

        matches = pattern.findall(data)
        if not matches:
            raise ValueError(f'Faield to import txt data: no matches for {data}')
        
        data = numpy.array(matches, dtype=dtype)  # parse floats first

        # Fill fields
        start = 0
        for semantic in self.layout.semantics:
            n = semantic.get_num_values()
            field_data = data[:, start:start+n].astype(semantic.format.numpy_base_type)
            if n == 1:
                field_data = field_data.ravel()
            try:
                self.set_field(semantic.get_name(), field_data)
            except Exception as e:
                raise ValueError(f'Failed to import {semantic}: {str(e)}') from e
            start += n

    def get_bytes(self):
        return self.data.tobytes()

    def __len__(self):
        return len(self.data)


class BufferElement:
    def __init__(self, buffer, index):
        self.buffer = buffer
        self.index = index
        self.layout = self.buffer.layout

    def get_bytes(self, semantic, return_buffer_semantic=False):
        if isinstance(semantic, AbstractSemantic):
            semantic = self.layout.get_element(semantic)
        byte_offset = self.index * semantic.stride
        data_bytes = self.buffer.data[semantic][byte_offset : byte_offset + semantic.stride]
        if not return_buffer_semantic:
            return data_bytes
        else:
            return data_bytes, semantic

    def set_bytes(self, semantic, data_bytes):
        if isinstance(semantic, AbstractSemantic):
            semantic = self.layout.get_element(semantic)
        byte_offset = self.index * semantic.stride
        self.buffer.data[semantic][byte_offset: byte_offset + semantic.stride] = data_bytes

    def get_value(self, semantic):
        if isinstance(semantic, AbstractSemantic):
            semantic = self.layout.get_element(semantic)
        data_bytes = self.get_bytes(semantic)
        return semantic.format.decoder(data_bytes).tolist()

    def set_value(self, semantic, value):
        if isinstance(semantic, AbstractSemantic):
            semantic = self.layout.get_element(semantic)
        self.set_bytes(semantic, semantic.format.encoder(value).tobytes())

    def get_all_bytes(self):
        data_bytes = bytearray()
        for semantic in self.layout.semantics:
            data_bytes.extend(self.get_bytes(semantic))
        return data_bytes


class ByteBuffer:
    def __init__(self, layout, data_bytes=None):
        self.layout = None
        self.data = {}
        self.num_elements = 0

        self.update_layout(layout)

        if data_bytes is not None:
            self.from_bytes(data_bytes)

    def validate(self):
        result = {}
        for semantic in self.layout.semantics:
            result[semantic] = len(self.data[semantic]) / semantic.stride
        if min(result.values()) != max(result.values()):
            result = ', '.join([f'{k.abstract}: {v}' for k, v in result.items()])
            raise ValueError(f'elements count mismatch in buffers: {result}')
        if len(self.layout.semantics) != len(self.data):
            raise ValueError(f'data structure must match buffer layout!')
        self.num_elements = int(min(result.values()))

    def update_layout(self, layout):
        self.layout = layout
        if len(self.data) != 0:
            self.validate()

    def from_bytes(self, data_bytes):
        if self.layout.force_stride:
            data_bytes.extend(bytearray((math.ceil(len(data_bytes) / self.layout.stride)) * self.layout.stride - len(data_bytes)))

        num_elements = len(data_bytes) / self.layout.stride
        if num_elements % 1 != 0:
            raise ValueError(f'buffer stride {self.layout.stride} must be multiplier of bytes len {len(data_bytes)}')
        num_elements = int(num_elements)

        self.data = {}
        for semantic in self.layout.semantics:
            self.data[semantic] = bytearray()

        byte_offset = 0
        for element_id in range(num_elements):
            for semantic in self.layout.semantics:
                self.data[semantic].extend(data_bytes[byte_offset:byte_offset+semantic.stride])
                byte_offset += semantic.stride

        if byte_offset != len(data_bytes):
            raise ValueError(f'layout mismatch: input ended at {byte_offset} instead of {len(data_bytes)}')

        self.validate()

    def get_element(self, index):
        return BufferElement(self, index)

    def extend(self, num_elements):
        if num_elements <= 0:
            raise ValueError(f'cannot extend buffer by {num_elements} elements')
        for semantic in self.layout.semantics:
            if semantic in self.data:
                self.data[semantic].extend(bytearray(num_elements * semantic.stride))
            else:
                self.data[semantic] = bytearray(num_elements * semantic.stride)
        self.validate()

    def get_fragment(self, offset, element_count):
        fragment = ByteBuffer(self.layout)
        for semantic in self.layout.semantics:
            byte_offset = offset * semantic.stride
            byte_count = element_count * semantic.stride
            fragment.data[semantic] = self.data[semantic][byte_offset:byte_offset+byte_count]
        fragment.validate()
        return fragment

    def import_buffer(self, src_byte_buffer, semantic_map=None, skip_missing=False):
        """
        Imports elements from source buffer based on their semantics
        Without 'semantic_map' provided creates new 'semantic_map' containing all source semantics
        Errors if any of 'semantic_map' elements is not found in src or dst buffers and 'skip_missing' is False
        """
        # Ensure equal number of elements in both buffers
        if src_byte_buffer.num_elements != self.num_elements:
            raise ValueError('source buffer len %d differs from destination buffer len %d' % (
                    src_byte_buffer.num_elements, self.num_elements))
        
        # Calculate semantic map
        semantic_map = self.map_semantics(src_byte_buffer, self, semantic_map=semantic_map, skip_missing=skip_missing)

        # Import data bytes
        for src_semantic, dst_semantic in semantic_map.items():
            if src_semantic.format == dst_semantic.format:
                self.data[dst_semantic] = src_byte_buffer.data[src_semantic]
            else:
                src_values = src_semantic.format.decoder(src_byte_buffer.data[src_semantic]).tolist()
                self.data[dst_semantic] = dst_semantic.format.encoder(src_values).tobytes()

        self.validate()

    def get_bytes(self, semantic=None):
        if semantic is None:
            data_bytes = bytearray()
            for element_id in range(self.num_elements):
                data_bytes.extend(self.get_element(element_id).get_all_bytes())
            return data_bytes
        else:
            if isinstance(semantic, AbstractSemantic):
                semantic = self.layout.get_element(semantic)
            return self.data[semantic]

    def get_values(self, semantic):
        if isinstance(semantic, AbstractSemantic):
            semantic = self.layout.get_element(semantic)
        data_bytes = self.get_bytes(semantic)
        return semantic.format.decoder(data_bytes).tolist()

    def set_bytes(self, semantic, data_bytes):
        if isinstance(semantic, AbstractSemantic):
            semantic = self.layout.get_element(semantic)
        self.data[semantic] = data_bytes
        self.validate()

    def set_values(self, semantic, values):
        if isinstance(semantic, AbstractSemantic):
            semantic = self.layout.get_element(semantic)
        self.set_bytes(semantic, semantic.format.encoder(values).tobytes())

    @staticmethod
    def map_semantics(src_byte_buffer, dst_byte_buffer, semantic_map=None, skip_missing=False):
        """
        
        """
        verified_semantic_map = {}
        if semantic_map is not None:
            # Semantic map may consist of AbstractSemantic instead of BufferSemantic, we need to convert it in this case
            # AbstractSemantic is independent of buffer specifics and contains only SemanticName and SemanticIndex
            # BufferSemantic wraps AbstractSemantic and describes where AbstractSemantic is located in given buffer
            for src_semantic, dst_semantic in semantic_map.items():
                # Ensure source semantic location in source buffer
                src_semantic = src_semantic
                if isinstance(src_semantic, AbstractSemantic):
                    src_semantic = src_byte_buffer.layout.get_element(src_semantic)
                if src_semantic not in src_byte_buffer.layout.semantics:
                    if not skip_missing:
                        raise ValueError(f'source buffer has no {src_semantic.abstract} semantic')
                    continue
                # Ensure destination semantic location in destination buffer
                dst_semantic = src_semantic
                if isinstance(src_semantic, AbstractSemantic):
                    dst_semantic = dst_byte_buffer.layout.get_element(dst_semantic)
                if dst_semantic not in dst_byte_buffer.layout.semantics:
                    if not skip_missing:
                        raise ValueError(f'destination buffer has no {dst_semantic.abstract} semantic')
                    continue
                # Add semantic to verified map
                verified_semantic_map[src_semantic] = dst_semantic
        else:
            # If there is no semantics map provided, map everything by default
            for src_semantic in src_byte_buffer.layout.semantics:
                # Locate matching semantic in destination buffer
                dst_semantic = dst_byte_buffer.layout.get_element(src_semantic.abstract)
                if dst_semantic is None:
                    if not skip_missing:
                        raise ValueError(f'destination buffer has no {src_semantic.abstract} semantic')
                    continue
                verified_semantic_map[src_semantic] = dst_semantic

        return verified_semantic_map


class IndexBuffer(ByteBuffer):
    def __init__(self, layout, data, load_indices=True):
        self.offset = None
        self.first_index = None
        self.index_count = None
        self.topology = None
        self.format = None
        self.faces = None

        if isinstance(data, io.IOBase):
            self.parse_format(data)
            if load_indices:
                self.parse_faces(data)
            super().__init__(layout)
        elif isinstance(data, bytearray):
            super().__init__(layout, data)
            self.bytes_to_faces()
        else:
            raise ValueError(f'unknown IB data format {data}')

    def parse_format(self, f):
        for line in map(str.strip, f):
            if line.startswith('byte offset:'):
                self.offset = int(line[13:])
            elif line.startswith('first index:'):
                self.first_index = int(line[13:])
            elif line.startswith('index count:'):
                self.index_count = int(line[13:])
            elif line.startswith('topology:'):
                self.topology = line[10:]
                if line != 'topology: trianglelist':
                    raise ValueError('"%s" is not yet supported' % line)
            elif line.startswith('format:'):
                dxgi_format = line[8:].replace('DXGI_FORMAT_', '')
                self.format = dxgi_format
            elif line == '':
                if any(x is None for x in [self.offset, self.topology, self.format]):
                    raise ValueError('failed to parse IB format')
                break

    def parse_faces(self, f):
        self.faces = []
        for line in map(str.strip, f):
            face = tuple(map(int, line.split()))
            assert (len(face) == 3)
            self.faces.append(face)
        if self.index_count:
            assert (len(self.faces) * 3 == self.index_count)
        else:
            self.index_count = len(self.faces) * 3

    def faces_to_bytes(self):
        indices = []
        for face in self.faces:
            assert (len(face) == 3)
            indices.extend(list(face))
        assert (len(indices) == self.index_count)
        data_bytes = self.layout.semantics[0].format.encoder(indices).tobytes()
        self.from_bytes(data_bytes)
        assert (self.num_elements * 3 == self.index_count)

    def bytes_to_faces(self):
        self.faces = []
        for element_id in range(self.num_elements):
            face = self.get_element(element_id).get_value(self.layout.semantics[0])
            self.faces.append(tuple(face))

    def get_bytes(self, semantic=None):
        if self.num_elements * 3 != self.index_count:
            self.faces_to_bytes()
        assert (self.num_elements * 3 == self.index_count)
        return super().get_bytes(semantic)
    
    def get_numpy_array(self):
        return numpy.frombuffer(self.get_bytes(), dtype=self.layout.semantics[0].get_numpy_type())

    def get_format(self):
        return self.layout.get_element(AbstractSemantic(Semantic.Index)).get_format()
