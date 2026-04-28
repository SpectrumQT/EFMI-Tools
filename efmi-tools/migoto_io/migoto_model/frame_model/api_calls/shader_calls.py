from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Type

from ...types import ShaderType
from ..registry import register_command
from ..commands import CommandCall
from ..calls import ShaderCall, Shader

if TYPE_CHECKING:
    from ..frame_model import DumpModel


@dataclass(repr=False)
class SetShader(CommandCall):
    shader_hash: str = field(metadata={"arg": "hash"})
    shader_pointer: str = field(metadata={"arg": "pComputeShader"})

    def execute(self, dump_model: "DumpModel", shader_call: "ShaderCall"):
        shader_call.shaders.append(Shader(
            hash=self.shader_hash,
            pointer=self.shader_pointer,
            type=self.shader_type,
        ))


# CSSetShader(pComputeShader:0x0000001C4D08B338, ppClassInstances:0x0000000000000000, NumClassInstances:0) hash=d24a7a3f4e50cffe
@register_command("CSSetShader")
@dataclass(repr=False)
class CSSetShader(SetShader):
    shader_type = ShaderType.Compute
    shader_pointer: str = field(metadata={"arg": "pComputeShader"})


# VSSetShader(pVertexShader:0x0000001D9F3F7278, ppClassInstances:0x0000000000000000, NumClassInstances:0) hash=751b2129bfaca0dd
@register_command("VSSetShader")
@dataclass(repr=False)
class VSSetShader(SetShader):
    shader_type = ShaderType.Vertex
    shader_pointer: str = field(metadata={"arg": "pVertexShader"})


# PSSetShader(pPixelShader:0x0000001C4D085BB8, ppClassInstances:0x0000000000000000, NumClassInstances:0) hash=02a98dbdeaa957c3
@register_command("PSSetShader")
@dataclass(repr=False)
class PSSetShader(SetShader):
    shader_type = ShaderType.Pixel
    shader_pointer: str = field(metadata={"arg": "pPixelShader"})