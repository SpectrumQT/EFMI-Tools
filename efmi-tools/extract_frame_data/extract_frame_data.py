import os
import sys
import time
import json
import shutil

from pathlib import Path
from typing import Dict
from dataclasses import dataclass
from collections import OrderedDict
from textwrap import dedent

from ..addon.exceptions import ConfigError

from ..migoto_io.blender_interface.utility import *
from ..migoto_io.blender_interface.collections import *
from ..migoto_io.blender_interface.objects import *

from ..migoto_io.data_model.dxgi_format import DXGIFormat
from ..migoto_io.data_model.byte_buffer import ByteBuffer, IndexBuffer, MigotoFmt, BufferLayout, BufferSemantic, AbstractSemantic, Semantic, NumpyBuffer
from ..migoto_io.data_model.numpy_mesh import NumpyMesh, GeometryMatcher, VertexGroupsMatcher

from ..migoto_io.dump_parser.filename_parser import ShaderType, SlotType, SlotId
from ..migoto_io.dump_parser.dump_parser import Dump
from ..migoto_io.dump_parser.resource_collector import Source, WrappedResource
from ..migoto_io.dump_parser.calls_collector import ShaderMap, Slot
from ..migoto_io.dump_parser.data_collector import DataMap, DataCollector

from ..data_models.data_model_efmi import DataModelEFMI


from .data_extractor import DataExtractor
from .shapekey_builder import ShapeKeyBuilder
from .component_builder import ComponentBuilder
from .output_builder import OutputBuilder, TextureFilter, ObjectData
from .lod_matcher import LODMatcher
from .metadata_format import read_metadata, ExtractedObjectComponentLOD


@dataclass
class Configuration:
    # output_path: str
    # dump_dir_path: str
    shader_data_pattern: Dict[str, ShaderMap]
    shader_resources: Dict[str, DataMap]


# In WuWa VB is dynamically calculated by dedicated compute shaders (aka Pose CS)
# So mesh is getting rendered via following chain:
#               BONES -v  COLOR+TEXCOORD -v
#   BLEND+NORM+POS -> Pose CS -> VB -> VS & PS -> RENDER_TARGET
#    SHAPEKEY_OFFSETS -^            IB -^    ^- Textures
#                ^- Shape Keys Application CS Chain
#   SHAPEKEY_BUFFERS -^
#
# So we can grab all relevant data in 3 steps:
#   1. Collect VS>PS calls from dump
#   2. Collect CS calls from dump that output VB to #1 calls (cs-u0 and cs-u1 to vb)
#   3. For each unique VB output (cs-u0 & cs-u1) from #2 calls:
#        3.1. [BLEND+NORM+POS] Collect CS calls from #2 with VB as output (cs-u0 & cs-u1)
#        3.2. [VERT_COLOR_GROUPS] Collect PS calls from dump with output to #3.1 calls (cs-u0 to cs-t3)
#        3.3. [COLOR+TEXCOORD+IB+Textures] Collect VS>PS calls from #1 with VB as input (vb from cs-u0 and cs-u1)
#
configuration = Configuration(
    # output_path=r'C:\Projects\Wuthering Waves\3DMIGOTO_DEV\!PROJECTS\Collect',
    # dump_dir_path=r'C:\Projects\Wuthering Waves\3DMIGOTO_DEV\FrameAnalysis-2024-06-14-120528',
    # dump_dir_path=r'C:\Projects\Wuthering Waves\3DMIGOTO_DEV\FrameAnalysis-2024-06-10-190045',
    shader_data_pattern={
        'SKELETON_CS_0': ShaderMap(ShaderType.Compute,
                                   inputs=[],
                                   outputs=[Slot('DRAW_VS_TARGET', ShaderType.Empty, SlotType.UAV, SlotId(0))]),
        # 'SHAPEKEY_CS_1': ShaderMap(ShaderType.Compute,
        #                            inputs=[Slot('SHAPEKEY_CS_0', ShaderType.Empty, SlotType.UAV, SlotId(1))],
        #                            outputs=[Slot('SHAPEKEY_CS_2', ShaderType.Empty, SlotType.UAV, SlotId(0))]),
        # 'SHAPEKEY_CS_2': ShaderMap(ShaderType.Compute,
        #                            inputs=[Slot('SHAPEKEY_CS_1', ShaderType.Empty, SlotType.UAV, SlotId(0))],
        #                            outputs=[Slot('DRAW_VS_DUMMY', ShaderType.Empty, SlotType.UAV, SlotId(0))]),
        'DRAW_VS_TARGET': ShaderMap(ShaderType.Vertex,
                             inputs=[Slot('SKELETON_CS_0', ShaderType.Empty, SlotType.Texture, SlotId(0)),],
                             outputs=[]),
        'DRAW_VS': ShaderMap(ShaderType.Vertex,
                             # Hack: When shader is short cirquited on itself, calls with listed input slots will be excluded from resulting branch
                             inputs=[],
                             # Hack: Short cirquit shader on itself to allow search of shaders without outputs
                             outputs=[Slot('DRAW_VS', ShaderType.Empty, SlotType.Texture, SlotId(0))],),
    },
    shader_resources={

        'IB_BUFFER_TXT_CHECK': DataMap([
                Source('DRAW_VS_TARGET', ShaderType.Empty, SlotType.IndexBuffer, file_ext='txt', ignore_missing=True)
            ]),
        'POSITION_BUFFER_CHECK': DataMap([
                Source('DRAW_VS_TARGET', ShaderType.Empty, SlotType.VertexBuffer, SlotId(0), ignore_missing=True),
            ]),
        
        'BONES_DATA_CHECK': DataMap([
                Source('DRAW_VS', ShaderType.Vertex, SlotType.Texture, SlotId(0), ignore_missing=True)
            ]),

        'IB_BUFFER_TXT_HASH': DataMap([
                Source('DRAW_VS', ShaderType.Empty, SlotType.IndexBuffer, file_ext='txt', ignore_missing=True)
            ]),
        'POSITION_BUFFER_HASH': DataMap([
                Source('DRAW_VS', ShaderType.Empty, SlotType.VertexBuffer, SlotId(0), ignore_missing=True),
            ]),

        'IB_BUFFER_TXT': DataMap([
                Source('DRAW_VS', ShaderType.Empty, SlotType.IndexBuffer, file_ext='txt', ignore_missing=True)
            ],
            BufferLayout([
                BufferSemantic(AbstractSemantic(Semantic.Index, 0), DXGIFormat.R16_UINT, stride=6),
            ])),
        'POSITION_BUFFER': DataMap([
                Source('DRAW_VS', ShaderType.Empty, SlotType.VertexBuffer, SlotId(0), ignore_missing=True),
            ],
            BufferLayout([
                BufferSemantic(AbstractSemantic(Semantic.Position, 0), DXGIFormat.R32G32B32_FLOAT, input_slot=0),
                BufferSemantic(AbstractSemantic(Semantic.EncodedData, 0), DXGIFormat.R32_UINT, input_slot=0),
            ])),
        'TEXCOORD_BUFFER': DataMap([
                Source('DRAW_VS', ShaderType.Empty, SlotType.VertexBuffer, SlotId(1), ignore_missing=True),
            ],
            BufferLayout([
                BufferSemantic(AbstractSemantic(Semantic.TexCoord, 0), DXGIFormat.R32G32_FLOAT, input_slot=1),
                BufferSemantic(AbstractSemantic(Semantic.Color, 0), DXGIFormat.R8G8B8A8_SNORM, input_slot=1),
            ])),
        'TEXCOORD_BUFFER_SINGLE': DataMap([
                Source('DRAW_VS', ShaderType.Empty, SlotType.VertexBuffer, SlotId(1), ignore_missing=True),
            ],
            BufferLayout([
                BufferSemantic(AbstractSemantic(Semantic.TexCoord, 0), DXGIFormat.R32G32_FLOAT, input_slot=1),
            ])),
        'TEXCOORD_BUFFER_DOUBLE': DataMap([
                Source('DRAW_VS', ShaderType.Empty, SlotType.VertexBuffer, SlotId(1), ignore_missing=True),
            ],
            BufferLayout([
                BufferSemantic(AbstractSemantic(Semantic.TexCoord, 0), DXGIFormat.R32G32_FLOAT, input_slot=1),
                BufferSemantic(AbstractSemantic(Semantic.TexCoord, 4), DXGIFormat.R32G32_FLOAT, input_slot=1),
            ])),
        'BLEND_BUFFER': DataMap([
                Source('DRAW_VS', ShaderType.Empty, SlotType.VertexBuffer, SlotId(2), ignore_missing=True),
            ],
            BufferLayout([
                BufferSemantic(AbstractSemantic(Semantic.Blendweight, 0), DXGIFormat.R16G16B16A16_UNORM, input_slot=2),
                BufferSemantic(AbstractSemantic(Semantic.Blendindices, 0), DXGIFormat.R8G8B8A8_UINT, input_slot=2),
            ], force_stride=True)),
        'BLEND_BUFFER_IDX_ONLY': DataMap([
                Source('DRAW_VS', ShaderType.Empty, SlotType.VertexBuffer, SlotId(2), ignore_missing=True),
            ],
            BufferLayout([
                BufferSemantic(AbstractSemantic(Semantic.Blendindices, 0), DXGIFormat.R8G8B8A8_UINT, input_slot=2),
            ], force_stride=True)),
        
        'TEXTURE_0': DataMap([Source('DRAW_VS', ShaderType.Pixel, SlotType.Texture, SlotId(0), ignore_missing=True)]),
        'TEXTURE_1': DataMap([Source('DRAW_VS', ShaderType.Pixel, SlotType.Texture, SlotId(1), ignore_missing=True)]),
        'TEXTURE_2': DataMap([Source('DRAW_VS', ShaderType.Pixel, SlotType.Texture, SlotId(2), ignore_missing=True)]),
        'TEXTURE_3': DataMap([Source('DRAW_VS', ShaderType.Pixel, SlotType.Texture, SlotId(3), ignore_missing=True)]),
        'TEXTURE_4': DataMap([Source('DRAW_VS', ShaderType.Pixel, SlotType.Texture, SlotId(4), ignore_missing=True)]),
        'TEXTURE_5': DataMap([Source('DRAW_VS', ShaderType.Pixel, SlotType.Texture, SlotId(5), ignore_missing=True)]),
        'TEXTURE_6': DataMap([Source('DRAW_VS', ShaderType.Pixel, SlotType.Texture, SlotId(6), ignore_missing=True)]),
        'TEXTURE_7': DataMap([Source('DRAW_VS', ShaderType.Pixel, SlotType.Texture, SlotId(7), ignore_missing=True)]),
        'TEXTURE_8': DataMap([Source('DRAW_VS', ShaderType.Pixel, SlotType.Texture, SlotId(8), ignore_missing=True)]),
        
    },
)


def write_objects(output_directory, objects: Dict[str, ObjectData], allow_missing_shapekeys = False):
    output_directory = Path(output_directory)

    output_directory.mkdir(parents=True, exist_ok=True)

    for object_hash, object_data in objects.items():
        object_name = object_hash
        
        if object_data.shapekeys.offsets_hash and not object_data.shapekeys.shapekey_offsets:
            if allow_missing_shapekeys:
                object_name += '_MISSING_SHAPEKEYS'
            else:
                continue

        object_directory = output_directory / object_name
        object_directory.mkdir(parents=True, exist_ok=True)

        textures = {}
        texture_usage = {}
        
        for component_id, component in enumerate(object_data.components):

            component_filename = f'Component {component_id}'

            # Write buffers
            with open(object_directory / f'{component_filename}.ib', "wb") as f:
                f.write(component.ib)
            with open(object_directory / f'{component_filename}.vb', "wb") as f:
                f.write(component.vb)
            with open(object_directory / f'{component_filename}.fmt', "w") as f:
                f.write(component.fmt)

            # Write textures
            texture_usage[component_filename] = OrderedDict()
            for texture in component.textures:

                if texture.hash not in textures:
                    textures[texture.hash] = {
                        'path': texture.path,
                        'components': []
                    }

                textures[texture.hash]['components'].append(str(component_id))

                if texture.get_slot() not in texture_usage[component_filename]:
                    texture_usage[component_filename][texture.get_slot()] = []

                shaders = '-'.join([shader.raw for shader in texture.shaders])
                texture_usage[component_filename][texture.get_slot()].append(f'{texture.hash}-{shaders}')
                
            texture_usage[component_filename] = OrderedDict(sorted(texture_usage[component_filename].items()))

        for texture_hash, texture in textures.items():
            path = Path(texture['path'])
            components = '-'.join(sorted(list(set(texture['components']))))
            shutil.copyfile(path, object_directory / f'Components-{components} t={texture_hash}{path.suffix}')
            
        with open(object_directory / f'TextureUsage.json', "w") as f:
            f.write(json.dumps(texture_usage, indent=4))

        with open(object_directory / f'Metadata.json', "w") as f:
            f.write(object_data.metadata)


def extract_frame_data(cfg, extract_lods=False):

    configuration_new = Configuration(
        shader_data_pattern={
            'SKELETON_CS_0': ShaderMap(
                ShaderType.Compute,
                inputs=[],
                outputs=[Slot('DRAW_VS', ShaderType.Empty, SlotType.UAV, SlotId(0))]
            ),
            'DRAW_VS': ShaderMap(
                ShaderType.Vertex,
                inputs=[Slot('SKELETON_CS_0', ShaderType.Empty, SlotType.Texture, SlotId(0)),],
                outputs=[]
            ),
        },
        shader_resources={
            'IB': DataMap([
                Source('DRAW_VS', ShaderType.Empty, SlotType.IndexBuffer, file_ext='buf', parse_header=True, ignore_missing=True)
            ]),
            'VB0': DataMap([
                Source('DRAW_VS', ShaderType.Empty, SlotType.VertexBuffer, SlotId(0), file_ext='buf', parse_header=True, ignore_missing=True),
            ]),
            'VB1': DataMap([
                Source('DRAW_VS', ShaderType.Empty, SlotType.VertexBuffer, SlotId(1), file_ext='buf', parse_header=True, ignore_missing=True),
            ]),
            'VB2': DataMap([
                Source('DRAW_VS', ShaderType.Empty, SlotType.VertexBuffer, SlotId(2), file_ext='buf', parse_header=True, ignore_missing=True),
            ]),
            'VB3': DataMap([
                Source('DRAW_VS', ShaderType.Empty, SlotType.VertexBuffer, SlotId(3), file_ext='buf', parse_header=True, ignore_missing=True),
            ]),
            'VB4': DataMap([
                Source('DRAW_VS', ShaderType.Empty, SlotType.VertexBuffer, SlotId(4), file_ext='buf', parse_header=True, ignore_missing=True),
            ]),
        },
    )

    if not extract_lods:
        for i in range(32):
            configuration_new.shader_resources[f'TEXTURE_{i}'] = DataMap([Source('DRAW_VS', ShaderType.Pixel, SlotType.Texture, SlotId(i), ignore_missing=True)])
        dump_path = resolve_path(cfg.frame_dump_folder)
    else:
        dump_path = resolve_path(cfg.lod_frame_dump_folder)
        
    start_time = time.time()

    if not dump_path.is_dir():
        raise ConfigError('frame_dump_folder', 'Specified dump folder does not exist!')
    if not Path(dump_path / 'log.txt').is_file():
        raise ConfigError('frame_dump_folder', 'Specified dump folder is missing log.txt file!')
    
    # Create data model of the frame dump
    dump = Dump(
        dump_directory=dump_path
    )

    # Get data view from dump data model
    frame_data = DataCollector(
        dump=dump,
        shader_data_pattern=configuration_new.shader_data_pattern,
        shader_resources=configuration_new.shader_resources
    )

    # Extract mesh objects data from data view
    data_extractor = DataExtractor(
        call_branches=frame_data.call_branches
    )

    # Build shape keys index from byte buffers
    shapekeys = ShapeKeyBuilder(
        shapekey_data=data_extractor.shape_key_data
    )

    # Build components from byte buffers
    component_builder = ComponentBuilder(
        output_vb_layout=None,
        shader_hashes=data_extractor.shader_hashes,
        shapekeys=shapekeys.shapekeys,
        draw_data=data_extractor.draw_data
    )

    # Build output data object
    output_builder = OutputBuilder(
        shapekeys=shapekeys.shapekeys,
        mesh_objects=component_builder.mesh_objects,
        texture_filter=TextureFilter(
            min_file_size=cfg.skip_small_textures_size*1024 if cfg.skip_small_textures else 0,
            exclude_extensions=['jpg'] if cfg.skip_jpg_textures else [],
            exclude_same_slot_hash_textures=cfg.skip_same_slot_hash_textures,
            exclude_hashes=['af26db30', '1320a071', '10d7937d', '87505b2b'] if cfg.skip_known_cubemap_textures else []
        )
    )

    if extract_lods:
        # full_model_path = Path(r"C:\Projects\XXMI\XXMI-Launcher\!RELEASES\1.9.6\XXMI Launcher\HIMI\Extracted Objects\ENDMIN_FULL")
        object_source_folder = resolve_path(cfg.object_source_folder)

        # lod_model_path = Path(r"C:\Projects\XXMI\XXMI-Launcher\!RELEASES\1.9.6\XXMI Launcher\HIMI\Extracted Objects\ENDMIN_LOD_OW")
    
        lod_matcher = LODMatcher(
            full_model_path=object_source_folder,
            lod_model_path=None,
            lod_objects=output_builder.objects,
            geo_matcher=GeometryMatcher(samples_count=500),
            vg_matcher=VertexGroupsMatcher(candidates_count=3),
        )

        lod_matcher.run()

        extracted_object = read_metadata(object_source_folder / 'Metadata.json')
        imported_lods_count = 0

        imported_objects = []

        if cfg.import_matched_lod_objects:
            model = DataModelEFMI()
            model.unpack_normal = True
    
        for component_id, component in enumerate(extracted_object.components):
            (lod_vb0_hash, vg_map, best_similarity) = lod_matcher.vg_maps.get(component.vb0_hash, (None, None, None))
            if lod_vb0_hash is None:
                continue
            if component.lods is None:
                component.lods = []
            if component.vb0_hash != lod_vb0_hash:
                imported_lods_count += 1

            if cfg.import_matched_lod_objects:
                # Import lod mesh for debug
                lod_name = f'LOD mesh {lod_vb0_hash}' if lod_vb0_hash != component.vb0_hash else '(full mesh used as LOD)'
                mesh = bpy.data.meshes.new(f'Component {component_id} {component.vb0_hash} {lod_name}')
                obj = bpy.data.objects.new(mesh.name, mesh)
                # global_matrix = axis_conversion(from_forward=axis_forward, from_up=axis_up).to_4x4()
                # obj.matrix_world = global_matrix
                mesh_name = lod_matcher.matched[component.vb0_hash]
                matched_mesh: NumpyMesh = lod_matcher.lod_meshes.get(mesh_name, None) or lod_matcher.full_meshes.get(mesh_name, None)
                model.set_data(obj, mesh, matched_mesh.index_buffer, matched_mesh.vertex_buffer, None, mirror_mesh=cfg.mirror_mesh, mesh_scale=1.00, mesh_rotation=(0, 0, 0), import_tangent_data_to_attribute=False)
                imported_objects.append(obj)

            if best_similarity < cfg.geo_matcher_error_threshold:
                raise ConfigError('lod_frame_dump_folder', dedent(f"""
                    Best matching LoD for Component {component_id} has {best_similarity:.2f}% similarity!
                    It is below configured {cfg.geo_matcher_error_threshold:.2f}% Geometry Matcher Error Threshold.
                    If it's not too far off, try to lower threshold. Otherwise either dump is missing some data or search engine fails to handle it.
                """))
            print(f'LOD Found: Component {component_id} {component.vb0_hash} matches LOD {lod_vb0_hash} ({best_similarity:.2f}% similarity)')

            # Skip LoD import if it already exists in Metadata.json
            if any(obj.vb0_hash == lod_vb0_hash for obj in component.lods):
                print(f'LOD {lod_vb0_hash} import skipped (already in Metdata.json)')
                continue

            component.lods.append(ExtractedObjectComponentLOD(vb0_hash=lod_vb0_hash, vg_map=vg_map))

        col = new_collection(f'{object_source_folder.stem} LoDs (for view only)')
        for obj in imported_objects:
            link_object_to_collection(obj, col)
            
        with open(object_source_folder / f'Metadata.json', 'w') as f:
            f.write(extracted_object.as_json())

        if imported_lods_count < len(extracted_object.components) / 2:
            bpy.context.window_manager.popup_menu(
                lambda self, context: self.layout.label(text=f'Imported LoDs count {imported_lods_count} is suspiciously low for {len(extracted_object.components)} components total. Please try to create another Open World dump with being further away from the character in case you get LoD issues with exported mod.'),
                title="LOD Import Complete",
                icon='WARNING'
            )
        else:
            bpy.context.window_manager.popup_menu(
                lambda self, context: self.layout.label(text=f'Successfully imported {imported_lods_count} LODs for {len(extracted_object.components)} components to Metadata.json ({len(extracted_object.components)-imported_lods_count} components seem to use full mesh as LOD).'),
                title="LOD Import Complete",
                icon='INFO'
            )

    if not extract_lods:
        write_objects(resolve_path(cfg.extract_output_folder), output_builder.objects, cfg.allow_missing_shapekeys)

    print(f"Execution time: %s seconds" % (time.time() - start_time))

    return output_builder


def get_dir_path():
    dir_path = ""

    if len(sys.argv) > 1:
        dir_path = sys.argv[1]

    if not os.path.exists(dir_path):
        print('Enter the name of frame dump folder:')
        dir_path = input()

    dir_path = os.path.abspath(dir_path)

    if not os.path.exists(dir_path):
        raise ValueError(f'Folder not found: {dir_path}!')
    if not os.path.isdir(dir_path):
        raise ValueError(f'Not a folder: {dir_path}!')

    return dir_path


if __name__ == "__main__":
    # try:
    extract_frame_data(configuration.dump_dir_path, configuration.output_path)
    # except Exception as e:
    #     print(f'Error: {e}')
    #     input()
