import re

from pathlib import Path
from dataclasses import dataclass, field
from typing import ClassVar, Match

from .helpers import AutoArgsMixin
from .types import DXGI_FORMAT, Topology, ShaderType, SlotType, ContaminationType


@dataclass
class MigotoResourceDescriptor(AutoArgsMixin):
    resource_hash: str | None = field(metadata={"arg": "resource_hash"})
    original_hash: str | None = field(metadata={"arg": "original_hash"})

    # Matches resource hash at filename start
    # 1dad3408.buf
    _FILENAME_PATTERN: ClassVar[re.Pattern] = re.compile(
        r"^(?P<resource_hash>[a-f0-9]+)",
        re.VERBOSE
    )

    @classmethod
    def parse_filename(cls, filename: str) -> Match[str]:
        """
        Parse a filename using the class's _FILENAME_PATTERN.

        Returns:
            - match: the compiled regex Match object

        Raises:
            ValueError: if the filename does not match the pattern
        """
        match: Match[str] | None = cls._FILENAME_PATTERN.match(filename)
        if not match:
            raise ValueError(f"Cannot parse filename: {filename}")
        return match

    @classmethod
    def from_filename(cls, filename: Path, return_match: bool = False) -> "MigotoResourceDescriptor | tuple[MigotoResourceDescriptor, Match[str]]":
        match = cls.parse_filename(filename.stem)
        arguments = match.groupdict()
        kwargs = cls.field_kwargs_from_dict(arguments)

        obj = cls(**kwargs)

        if return_match:
            return obj, match
        return obj


@dataclass
class MigotoTextureDescriptor(MigotoResourceDescriptor):
    data_format: DXGI_FORMAT | None = field(metadata={"arg": "data_format"})

    # Matches resource hash and texture format
    # 86074444-BC7_UNORM.jpg
    _FILENAME_PATTERN: ClassVar[re.Pattern] = re.compile(
        r"^(?P<resource_hash>[a-f0-9]+)-(?P<data_format>[A-Z0-9_]+)?",
        re.VERBOSE
    )


@dataclass
class MigotoIndexBufferDescriptor(MigotoResourceDescriptor):
    data_format: DXGI_FORMAT | None = field(metadata={"arg": "ib_format"})
    topology: Topology | None = field(metadata={"arg": "topology"})
    byte_offset: int | None = field(metadata={"arg": "byte_offset"})
    first_index: int | None = field(metadata={"arg": "first_index"})
    index_count: int | None = field(metadata={"arg": "index_count"})

    # Matches resource hash and inline IB fmt
    # 4a0facf9-ib-format=R16_UINT-topology=trianglelist-offset=523840-first=1608-count=54.txt
    _FILENAME_PATTERN: ClassVar[re.Pattern] = re.compile(
        r"""
        ^(?P<resource_hash>[a-f0-9]+)
        -ib
        (-format=(?P<ib_format>[A-Z0-9_]+))?
        (-topology=(?P<topology>[a-z]+))?
        (-offset=(?P<byte_offset>\d+))?
        (-first=(?P<first_index>\d+))?
        (-count=(?P<index_count>\d+))?
        """,
        re.VERBOSE
    )


@dataclass
class MigotoVertexBufferDescriptor(MigotoResourceDescriptor):
    slot_id: int = field(metadata={"arg": "slot_id"})
    layout_hash: str | None = field(metadata={"arg": "layout_hash"})
    topology: Topology | None = field(metadata={"arg": "topology"})
    byte_offset: int | None = field(metadata={"arg": "byte_offset"})
    stride: int | None = field(metadata={"arg": "stride"})
    first_vertex: int | None = field(metadata={"arg": "first_vertex"})
    vertex_count: int | None = field(metadata={"arg": "vertex_count"})
    first_instance: int | None = field(metadata={"arg": "first_instance"})
    instance_count: int | None = field(metadata={"arg": "instance_count"})

    # Matches resource hash and inline VB fmt
    # 4a0facf9-vb0-layout=63cc2401-topology=trianglelist-offset=48-stride=76-first=1072-count=36-first_inst=1-inst_count=2.txt
    _FILENAME_PATTERN: ClassVar[re.Pattern] = re.compile(
        r"""
        ^(?P<resource_hash>[a-f0-9]+)
        -vb(?P<slot_id>\d+)
        (-layout=(?P<layout_hash>[a-f0-9]+))?
        (-topology=(?P<topology>[a-z]+))?
        (-offset=(?P<byte_offset>\d+))?
        (-stride=(?P<stride>\d+))?
        (-first=(?P<first_vertex>\d+))?
        (-count=(?P<vertex_count>\d+))?
        (-first_inst=(?P<first_instance>\d+))?
        (-inst_count=(?P<instance_count>\d+))?
        """,
        re.VERBOSE
    )


@dataclass
class MigotoBufferDescriptor(MigotoResourceDescriptor):
    call_id: int = field(metadata={"arg": "call_id"})
    shader_type: ShaderType | None = field(metadata={"arg": "shader_type"})
    slot_type: SlotType = field(metadata={"arg": "slot_type"})
    slot_id: int | None = field(metadata={"arg": "slot_id"})
    contamination: ContaminationType | None = field(metadata={"arg": "contamination"})

    shaders: dict[ShaderType, str] = field(default_factory=dict)

    # Matches buffer usage inline format
    # 000001-cs-cb0=a517561d-cs=1f543271de52fe8d.buf
    # 000012-ib=536e6e9d-vs=b960839f9608d37c-ps=7b067f2b06883a0c.buf
    # 000002-vb0=a121f12b-vs=14c64d32eecc4287-ps=2b58d932af001aef.jpg
    # 000002-ps-t0=!S!=d0223636-vs=14c64d32eecc4287-ps=2b58d932af001aef.jpg
    # 000105-vb0=ab3c72c3(9a09f1f0)-vs=d9d6448a7b62687e-ps=885e3a3bb9607c6b.buf
    _FILENAME_PATTERN: ClassVar[re.Pattern] = re.compile(
        r"""
        ^(?P<call_id>\d+)-
        ((?P<shader_type>vs|ps|cs|gs|hs|ds)-)?(?P<slot_type>cb|t|vb|o|u|ib)(?P<slot_id>\d+)?  # Slot: vs-cb0, vb1, ib
        (=!(?P<contamination>[MUCS])!)?  # Contamination type: !M!
        (=(?P<resource_hash>[a-f0-9]+))?  # Hash: a517561d
        (\((?P<original_hash>[a-f0-9]+)\))?  # Original hash: (9a09f1f0)
        """,
        re.VERBOSE
    )
    _SHADER_PATTERN: ClassVar[re.Pattern] = re.compile(r"-(?P<shader_type>vs|ps|cs|gs|hs|ds)=(?P<hash>[a-f0-9]+)")

    @classmethod
    def parse_shaders(cls, string: str) -> dict[ShaderType, str]:
        shaders = {}
        for shader_match in cls._SHADER_PATTERN.finditer(string):
            shaders[ShaderType(shader_match.group("shader_type"))] = shader_match.group("hash")
        return shaders

    @classmethod
    def from_filename(cls, filename: Path, return_match: bool = False) -> "MigotoBufferDescriptor | tuple[MigotoBufferDescriptor, Match[str]]":
        obj, match = super().from_filename(filename, return_match=True)

        # Parse shaders from the remaining part of the filename
        rest = filename.stem[match.end():]
        obj.shaders = cls.parse_shaders(rest)

        if return_match:
            return obj, match
        return obj
