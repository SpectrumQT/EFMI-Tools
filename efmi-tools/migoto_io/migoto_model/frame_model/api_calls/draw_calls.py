from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ...types import ShaderType
from ..registry import register_command
from ..commands import CommandCall

if TYPE_CHECKING:
    from ..frame_model import DumpModel
    from ..calls import ShaderCall



@dataclass(repr=False)
class DrawCall(CommandCall):

    def execute(self, dump_model: "DumpModel", shader_call: "ShaderCall"):
        shader_call.draw_call = self


# Draw(VertexCount:4, StartVertexLocation:0)
@register_command("Draw")
@dataclass(repr=False)
class Draw(DrawCall):
    vertex_count: int = field(metadata={"arg": "VertexCount"})
    first_vertex: int = field(metadata={"arg": "StartVertexLocation"})


# DrawInstanced(VertexCountPerInstance:3, InstanceCount:1, StartVertexLocation:0, StartInstanceLocation:0)
@register_command("DrawInstanced")
@dataclass(repr=False)
class DrawInstanced(DrawCall):
    vertex_count: int = field(metadata={"arg": "VertexCountPerInstance"})
    first_vertex: int = field(metadata={"arg": "StartVertexLocation"})
    instance_count: int = field(metadata={"arg": "InstanceCount"})
    first_instance: int = field(metadata={"arg": "StartInstanceLocation"})


# DrawIndexed(IndexCount:36, StartIndexLocation:0, BaseVertexLocation:0)
@register_command("DrawIndexed")
@dataclass(repr=False)
class DrawIndexed(DrawCall):
    index_count: int = field(metadata={"arg": "IndexCount"})
    first_index: int = field(metadata={"arg": "StartIndexLocation"})
    first_vertex: int = field(metadata={"arg": "BaseVertexLocation"})


# DrawIndexedInstanced(IndexCountPerInstance:19920, InstanceCount:1, StartIndexLocation:0, BaseVertexLocation:0, StartInstanceLocation:0)
@register_command("DrawIndexedInstanced")
@dataclass(repr=False)
class DrawIndexedInstanced(DrawCall):
    index_count: int = field(metadata={"arg": "IndexCountPerInstance"})
    first_index: int = field(metadata={"arg": "StartIndexLocation"})
    first_vertex: int = field(metadata={"arg": "BaseVertexLocation"})
    instance_count: int = field(metadata={"arg": "InstanceCount"})
    first_instance: int = field(metadata={"arg": "StartInstanceLocation"})


# Dispatch(ThreadGroupCountX:480, ThreadGroupCountY:200, ThreadGroupCountZ:1)
@register_command("Dispatch")
@dataclass(repr=False)
class Dispatch(DrawCall):
    shader_type = ShaderType.Compute
    thread_group_count_x: int = field(metadata={"arg": "ThreadGroupCountX"})
    thread_group_count_y: int = field(metadata={"arg": "ThreadGroupCountY"})
    thread_group_count_z: int = field(metadata={"arg": "ThreadGroupCountZ"})
