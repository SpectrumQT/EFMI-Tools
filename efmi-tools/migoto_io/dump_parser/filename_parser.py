
import os
import shutil
import hashlib
import re

from enum import Enum, auto
from typing import Optional, Union, List, Dict, Set
from textwrap import dedent
from pathlib import Path

from ..data_model.byte_buffer import ByteBuffer, BufferLayout, IndexBuffer, MigotoFormat, NumpyBuffer

from .dict_filter import DictFilter, FilterCondition, Filter


class ShaderType(Enum):
    Empty = 'null'
    Compute = 'cs'
    Pixel = 'ps'
    Vertex = 'vs'
    Geometry = 'gs'
    Hull = 'hs'
    Domain = 'ds'


class BufferType(Enum):
    Blend = auto()
    Normal = auto()
    Position = auto()
    TexCoord = auto()
    ShapeKeyGroup = auto()
    ShapeKeyVertId = auto()
    ShapeKeyColor = auto()


class SlotType(Enum):
    ConstantBuffer = 'cb'
    IndexBuffer = 'ib'
    VertexBuffer = 'vb'
    Texture = 't'
    RenderTarget = 'o'
    UAV = 'u'


def SlotId(slot_id):
    return int(slot_id)


shader_type_codepage = {
    'cs': ShaderType.Compute,
    'ps': ShaderType.Pixel,
    'vs': ShaderType.Vertex,
    'gs': ShaderType.Geometry,
    'hs': ShaderType.Hull,
    'ds': ShaderType.Domain,
}

slot_type_codepage = {
    'o': SlotType.RenderTarget,
    't': SlotType.Texture,
    'u': SlotType.UAV,
    'cb': SlotType.ConstantBuffer,
    'ib': SlotType.IndexBuffer,
    'vb': SlotType.VertexBuffer,
}


class ShaderRef:
    def __init__(self, raw_shader_ref):
        self.raw = raw_shader_ref
        self.type = None
        self.hash = None
        self.parse_raw_ref()
        self.validate()

    def validate(self):
        if self.type is None:
            raise ValueError(f'Failed to parse shader ref "{self.raw}": shader type not detected!')
        if self.hash is None:
            raise ValueError(f'Failed to parse shader ref "{self.raw}": shader hash not detected!')

    def parse_raw_ref(self):
        result = self.raw.split('=')
        if len(result) != 2:
            return
        self.parse_raw_shader_ref(result[0])
        self.hash = result[1]

    def parse_raw_shader_ref(self, raw_shader_ref):
        self.type = shader_type_codepage.get(raw_shader_ref, None)


class ResourceData:
    def __init__(self, file_path):
        self.path = file_path
        self.header = None
        self.bytes = None
        self.len = None
        self.sha256 = None
        self.header_sha256 = None

    def load_header(self):
        with open(Path(self.path).with_suffix('.txt'), 'r') as f:
            self.header = MigotoFormat.extract_txt_file_fmt_text(f)

    def unload_header(self):
        self.header = None

    def update_header_hash(self):
        if self.header is None:
            raise ValueError("Failed to update resource header hash: file not loaded!")
        self.header_sha256 = hashlib.sha256(self.header.encode()).hexdigest()

    def load(self):
        with open(self.path, "rb") as f:
            self.bytes = bytearray(f.read())

    def unload(self):
        self.bytes = None

    def update_hash(self):
        if self.bytes is None:
            raise ValueError("Failed to update resource hash: file not loaded!")
        self.sha256 = hashlib.sha256(self.bytes).hexdigest()

    def update_len(self):
        if self.bytes is None:
            raise ValueError("Failed to update resource data len: file not loaded!")
        self.len = len(self.bytes)

    def get_header(self):
        is_unloaded = self.header is None
        if is_unloaded:
            self.load_header()
        header = self.header
        if is_unloaded:
            self.unload_header()
        return header


class ResourceDescriptor:
    def __init__(self, resource_file_path, calculate_sha256=False):
        self.path = resource_file_path
        self.raw = os.path.basename(resource_file_path)
        self.marked = False
        self.call: CallDescriptor = None
        self.call_id: str = None
        self.ext: str = None
        self.slot_type: SlotType = None
        self.slot_id: int = None
        self.slot_shader_type: ShaderType = None
        self.hash: str = None
        self.old_hash: str = None
        self.data: ResourceData = ResourceData(self.path)
        self.shaders = []
        if calculate_sha256:
            self.hash_data()
        self.parse_raw_call()
        self.validate()

    def __repr__(self):
        return self.raw

    def validate(self):
        if self.call_id is None:
            raise ValueError(f'Failed to parse raw descriptor "{self.raw}": no call id detected!')
        if self.slot_type is None:
            raise ValueError(f'Failed to parse raw descriptor "{self.raw}": slot type not detected!')
        if len(self.shaders) == 0:
            raise ValueError(f'Failed to parse raw descriptor "{self.raw}": no shader refs detected!')

    def get_sha256(self):
        is_unloaded = self.data.sha256 is None
        if is_unloaded:
            self.data.load()
            self.data.update_hash()
            self.data.update_len()
        sha256 = self.data.sha256
        if is_unloaded:
            self.data.unload()
        return sha256

    def get_header_sha256(self):
        is_unloaded = self.data.header_sha256 is None
        if is_unloaded:
            self.data.load_header()
            self.data.update_header_hash()
        sha256 = self.data.header_sha256
        if is_unloaded:
            self.data.unload_header()
        return sha256
    
    def get_len(self):
        is_unloaded = self.data.len is None
        if is_unloaded:
            self.data.load()
            self.data.update_hash()
            self.data.update_len()
        data_len = self.data.len
        if is_unloaded:
            self.data.unload()
        return data_len

    def hash_data(self):
        self.data.load()
        self.data.update_hash()
        self.data.update_len()
        self.data.unload()
    
    def get_bytes(self):
        is_unloaded = self.data.bytes is None
        if is_unloaded:
            self.data.load()
        data_bytes = self.data.bytes
        if is_unloaded:
            self.data.unload()
        return data_bytes
    
    def get_slot(self):
        return f'{self.slot_shader_type.value}-{self.slot_type.value}{self.slot_id}'
    
    def get_slot_hash(self):
        return f'{self.get_slot()}-{self.hash}'

    def parse_raw_call(self):
        raw_call = self.raw
        # Process '!U!' mark
        if raw_call.find('!U!') != -1:
            self.marked = True
            raw_call = raw_call.replace('!U!=', '')
        # Match call id
        call_id_pattern = re.compile(r'^(\d+)-(.*)\.([a-z0-9]+)')
        result = call_id_pattern.findall(raw_call)
        # Return if call id not found
        if len(result) != 1:
            return
        result = result[0]
        if len(result) != 3:
            return
        # Store results
        call_id = result[0]
        raw_refs = result[1]
        ext = result[2]
        # Match shader refs
        shaders_pattern = re.compile(r'-([a-z]s=[a-f0-9]+)')
        raw_shaders_refs = shaders_pattern.findall(raw_refs)
        # Return if no shader refs found
        if len(raw_shaders_refs) < 1:
            return
        # Remove shaders refs from the raw string
        # Only resource ref should be left in raw string at this point
        raw_resource_ref = re.sub(shaders_pattern, '', raw_refs)

        self.call_id = call_id
        self.ext = ext
        self.parse_raw_resource_ref(raw_resource_ref)
        self.parse_raw_shader_refs(raw_shaders_refs)

    def parse_raw_resource_ref(self, raw_resource_ref):
        result = raw_resource_ref.split('=')

        if len(result) == 0:
            return

        if len(result) == 2:
            raw_hash = result[1]
            # Handle `texture_hash = 1` 3dm setting, resulting in names like `000003-ps-t1=0dbc4afc(5e9494f3)-vs=2fb5a3f559d5a6f9-ps=561bcd63f5b5531a`
            hashes = raw_hash.split('(')
            # Actual hash
            self.hash = hashes[0]
            # Hash that texture would have without `texture_hash = 1` enabled
            if len(hashes) > 1:
                self.old_hash = hashes[1].replace(')', '')
        else:
            self.hash = None

        resource_desc = result[0].split('-')

        if len(resource_desc) == 1:
            self.parse_raw_slot_ref(resource_desc[0], None)
        else:
            self.parse_raw_slot_ref(resource_desc[1], resource_desc[0])

    def parse_raw_slot_ref(self, raw_slot_ref, raw_shader_type):
        slot_ref_pattern = re.compile(r'^([a-z]+)([0-9]+)?')
        result = slot_ref_pattern.findall(raw_slot_ref)
        if len(result) != 1:
            return
        result = result[0]
        self.slot_type = slot_type_codepage.get(result[0], None)
        if self.slot_type is None:
            raise ValueError(f'Failed to parse slot ref "{raw_slot_ref}": slot type not recognized!')
        if len(result) == 2 and result[1] != '':
            self.slot_id = int(result[1])
        if raw_shader_type is not None:
            self.slot_shader_type = shader_type_codepage.get(raw_shader_type, None)
            if self.slot_shader_type is None:
                raise ValueError(f'Failed to parse slot shader type "{raw_shader_type}": shader type not recognized!')

    def parse_raw_shader_refs(self, raw_shader_refs):
        for raw_shader_ref in raw_shader_refs:
            self.shaders.append(ShaderRef(raw_shader_ref))

    def copy_file(self, dest_path):
        shutil.copyfile(self.path, dest_path)


class ResourceConflict(Enum):
    OldHash = ('old_hashes', 'old_hash')
    SlotType = ('slot_types', 'slot_type')
    SlotId = ('slot_ids', 'slot_id')
    SlotShaderType = ('slot_shader_types', 'slot_shader_type')


class WrappedResource:
    def __init__(self, descriptor: ResourceDescriptor, load_header=False):

        self.ext: str = descriptor.ext
        self.hash: str = descriptor.hash

        self.old_hashes: Set[str] = set([])
        self.slot_types: Set[SlotType] = set([])
        self.slot_ids: Set[int] = set([])
        self.slot_shader_types: Set[ShaderType] = set([])

        self.data: ResourceData = descriptor.data
        self.buffer: Optional[NumpyBuffer] = None
        
        self.formats: Dict[str, MigotoFormat] = {}
        self.call_formats: Dict[str, MigotoFormat] = {}

        self.descriptors = []

        self.bind_descriptor(descriptor, allow_conflicts=[
            ResourceConflict.OldHash, ResourceConflict.SlotId, ResourceConflict.SlotShaderType, ResourceConflict.SlotType
        ], load_header=load_header)

    def bind_descriptor(self, descriptor: ResourceDescriptor, allow_conflicts: List[ResourceConflict], error_on_conflict = False, load_header = False):
        # Add values
        for c in ResourceConflict:
            w_res_prop_name, res_des_prop_name = c.value
            current_values_set = getattr(self, w_res_prop_name)
            new_value = getattr(descriptor, res_des_prop_name)
            if new_value not in current_values_set:
                if c not in allow_conflicts:
                    msg = dedent(f"""
                        Failed to bind descriptor to wrapped resource due to {c.name} conflict:
                        Current={current_values_set} vs New={new_value}
                        WrappedResource={self}
                        ResourceDescriptor={descriptor}
                    """).strip()
                    if error_on_conflict:
                        raise ValueError(msg)
                    else:
                        print(msg)
                        continue
                current_values_set.add(new_value)
                # setattr(self, c.value, new_value)

            self.descriptors.append(descriptor)
        # Load header
        if load_header:
            if self.ext not in ['txt', 'buf']:
                raise ValueError(f'Header loading is not supported for `.{self.ext}` resource {self}')
            # Load header from the file, for .buf it'll be loaded from .txt companion
            descriptor.data.load_header()
            fmt_text = descriptor.data.get_header()
            self.call_formats[descriptor.call_id] = fmt_text
            self.formats[fmt_text] = MigotoFormat.from_fmt_text(fmt_text)

    def load_buffer(self, layout: BufferLayout):
        if self.ext == 'txt':
            raise NotImplementedError
        if self.ext != 'buf':
            raise ValueError(f'Buffer loading is not supported for `.{self.ext}` resource {self}')
        with open(self.data.path, 'rb') as f:
            self.buffer = NumpyBuffer(layout)
            self.buffer.import_raw_data(f.read())

    def get_format(self, call_id = None, header_fmt_text= None):
        if call_id:
            return self.formats[self.call_formats[call_id]]
        if header_fmt_text:
            return self.formats[header_fmt_text]
        raise ValueError
            
    def __str__(self):
        return f'{self.ext, self.hash, self.old_hashes, self.slot_types, self.slot_ids, self.slot_shader_types, self.data.path}'


class CallDescriptor:
    def __init__(self, call_id):
        self.id = call_id
        self.parameters = {}
        self.shaders = {}
        self.resources = {}

    def import_resource_descriptor(self, resource_descriptor):
        if resource_descriptor.call_id != self.id:
            raise ValueError(f'Failed to import resource descriptor {resource_descriptor.raw}: call id mismatch!')
        if resource_descriptor.ext == 'txt':
            return
        for shader in resource_descriptor.shaders:
            self.shaders[shader.raw] = shader

        self.resources[resource_descriptor.raw] = resource_descriptor

    def hash_resources(self):
        for resource in self.resources:
            resource.hash_data()

    def get_filtered_resources(self, filter_attributes):
        resource_filter = Filter(
            condition=FilterCondition.AND,
            attributes_condition=FilterCondition.AND,
            attributes=filter_attributes,
            dictionaries_condition=FilterCondition.AND,
            dictionaries=[
                self.resources
            ]
        )
        return DictFilter(resource_filter).filtered_dict

    def get_filtered_resource(self, filter_attributes):
        result = self.get_filtered_resources(filter_attributes)
        if len(result) == 1:
            return next(iter(result.values()))
        elif len(result) == 0:
            return None
        else:
            raise ValueError(f'Found more than 1 resource with provided attributes!')

    def __repr__(self):
        return f'{self.id}, {", ".join(self.shaders.keys())}'