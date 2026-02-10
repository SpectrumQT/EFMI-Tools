import time
import shutil

from typing import List, Dict, Union
from dataclasses import dataclass, field

from ..addon.exceptions import ConfigError

from ..migoto_io.blender_interface.utility import *
from ..migoto_io.blender_interface.collections import *
from ..migoto_io.blender_interface.objects import *
from ..migoto_io.blender_interface.mesh import *
from ..migoto_io.data_model.dxgi_format import DXGIFormat
from ..migoto_io.data_model.byte_buffer import NumpyBuffer, MigotoFmt, BufferLayout, BufferSemantic, Semantic, AbstractSemantic
from ..migoto_io.data_model.data_model import DataModel

from ..extract_frame_data.metadata_format import read_metadata, ExtractedObject

from .object_merger import ObjectMerger, SkeletonType, MergedObject, MergedObjectShapeKeys
from .metadata_collector import Version, ModInfo
from .texture_collector import Texture, get_textures
from .ini_maker import IniMaker

from ..data_models.data_model_efmi import DataModelEFMI

class Fatal(Exception): pass


data_models: Dict[str, DataModel] = {
    'EFMI': DataModelEFMI(),
}


# TODO: Add support of export of unhandled semantics from vertex attributes
class ModExporter:
    extracted_object: ExtractedObject
    skeleton_type: SkeletonType
    merged_object: MergedObject
    buffers: Dict[str, NumpyBuffer]
    textures: List[Texture]
    ini: IniMaker

    def __init__(self, context, cfg, excluded_buffers: List[str]):
        self.context = context
        self.cfg = cfg
        self.excluded_buffers = excluded_buffers

        self.object_source_folder = resolve_path(cfg.object_source_folder)
        self.mod_output_folder = resolve_path(cfg.mod_output_folder)
        self.meshes_path = self.mod_output_folder / 'Meshes'
        self.meshes_path.mkdir(parents=True, exist_ok=True)
        self.textures_path = self.mod_output_folder / 'Textures'
        self.textures_path.mkdir(parents=True, exist_ok=True)
        self.local_mod_logo_path = self.textures_path / 'Logo.dds'

        self.buffers = {}
        self.textures =  []

    def export_mod(self):
    
        self.verify_config()

        start_time = time.time()
        print(f"Mod export started for '{self.cfg.component_collection.name}' object")

        if self.cfg.custom_template_live_update:
            self.cfg.partial_export = False
            self.cfg.write_ini = True

        try:
            self.extracted_object = read_metadata(self.object_source_folder / 'Metadata.json')
        except FileNotFoundError:
            raise ConfigError('object_source_folder', 'Specified folder is missing Metadata.json!')
        except Exception as e:
            raise ConfigError('object_source_folder', f'Failed to load Metadata.json:\n{e}')
        
        lods_count = sum([len(component.lods) for component in self.extracted_object.components])
        if lods_count == 0 and not self.cfg.allow_export_without_lods:
            raise ConfigError('object_source_folder', """
                No LODs found for object's components in Metadata.json!
                Mod exported without LOD data will not work in open world.
                Switch toolkit to "Import LoDs From Dump" mode and feed it with open world dump.
                Please do note that to dump character LODs correctly you should switch to another character and take a few steps back.
            """)
        
        self.cfg.mod_skeleton_type = 'COMPONENT'
        self.skeleton_type = SkeletonType.Merged if self.cfg.mod_skeleton_type == 'MERGED' else SkeletonType.PerComponent

        self.merged_object = MergedObject(None, None, [], MergedObjectShapeKeys(), self.skeleton_type)

        user_context = get_user_context(self.context)

        for component_id, component in enumerate(self.extracted_object.components):
            index_count = 0
            vertex_count = 0
            shapekeys_vertex_count = 0

            if component.lods is None:
                component.lods = []

            try:
                merged_object = self.build_merged_object(component_id)
            except ConfigError as e:
                raise e
            except Exception as e:
                raise ConfigError('component_collection', f'Failed to create merged object from collection:\n{e}')
            
            self.merged_object.components += merged_object.components

            try:
                self.build_data_buffers(merged_object, component_id)
            except Exception as e:
                raise e
            finally:
                if self.cfg.remove_temp_object and merged_object.object is not None:
                    remove_mesh(merged_object.object.data)
                set_user_context(self.context, user_context)

            index_count += merged_object.index_count
            vertex_count += merged_object.vertex_count
            # shapekeys_vertex_count += merged_object.shapekeys.vertex_count

        self.merged_object.index_count = index_count
        self.merged_object.vertex_count = vertex_count
        # self.merged_object.shapekeys.vertex_count = shapekeys_vertex_count

        if not self.cfg.partial_export:
            self.textures = get_textures(self.object_source_folder, ['af26db30', '1320a071', '10d7937d', '87505b2b'] if self.cfg.skip_known_cubemap_textures else [])

            if self.cfg.write_ini:
                try:
                    self.build_mod_ini()
                except FileNotFoundError:
                    raise ConfigError('custom_template_source', f'Specified custom template file not found!')
                except Exception as e:
                    raise ConfigError('use_custom_template', f'Failed to build mod.ini from ini template:\n{e}')

        if self.cfg.custom_template_live_update:
            print(f'Total live ini template initialization time: {time.time() - start_time :.3f}s')
            return

        try:
            self.write_files()
        except Exception as e:
            raise ConfigError('mod_output_folder', f'Failed to write files to mod folder:\n{e}')
            
        self.buffers = {}
        self.textures = []

        print(f'Total mod export time: {time.time() - start_time :.3f}s')

    def verify_config(self):
        if self.cfg.component_collection is None:
            raise ConfigError('component_collection', f'Components collection is not specified!')
        if self.cfg.component_collection not in list(get_scene_collections()):
            raise ConfigError('component_collection', f'Collection "{self.cfg.component_collection.name}" is not a member of "Scene Collection"!')

    def build_merged_object(self, component_id = -1):
        start_time = time.time()
        object_merger = ObjectMerger(
            extracted_object=self.extracted_object,
            component_id=component_id,
            ignore_nested_collections=self.cfg.ignore_nested_collections,
            ignore_hidden_collections=self.cfg.ignore_hidden_collections,
            ignore_hidden_objects=self.cfg.ignore_hidden_objects,
            ignore_muted_shape_keys=self.cfg.ignore_muted_shape_keys,
            apply_modifiers=self.cfg.apply_all_modifiers,
            context=self.context,
            collection=self.cfg.component_collection,
            skeleton_type=self.skeleton_type,
            mesh_scale=1.00,
            mesh_rotation=(0, 0, 0),
            add_missing_vertex_groups=self.cfg.add_missing_vertex_groups,
            allow_empty_components=True,
        )
        print(f'Merged object build time: {time.time() - start_time :.3f}s ({self.merged_object.vertex_count} vertices, {self.merged_object.index_count} indices)')
        return object_merger.merged_object

    def build_data_buffers(self, merged_object: MergedObject, component_id = -1):
        start_time = time.time()

        global data_models
        data_model = data_models['EFMI']

        buffers_format: Dict[str, BufferLayout] = {}
        if self.extracted_object.export_format is not None and len(self.extracted_object.export_format) > 0:
            for buffer_name, buffer_layout in self.extracted_object.export_format.items():
                buffers_format[buffer_name] = buffer_layout.get_layout()

        index_layout = None
        if merged_object.object is not None and len(merged_object.object.vertex_groups) > 256:
            index_layout = []
            for component in merged_object.components:
                index_layout.append(component.index_count)

        fmt_path = self.object_source_folder / f'Component {component_id}.fmt'
        with open(fmt_path, 'r') as fmt:
            migoto_fmt = MigotoFmt(fmt)
            buffers_format[f'Component{component_id}_IB'] = migoto_fmt.ib_layout
            for buffer_semantic in migoto_fmt.vb_layout.semantics:
                buffer_name = f'Component{component_id}_VB{buffer_semantic.input_slot}'
                layout = buffers_format.get(buffer_name, BufferLayout([]))
                if not layout.semantics:
                    buffers_format[buffer_name] = layout
                layout.add_element(BufferSemantic(buffer_semantic.abstract, buffer_semantic.format))
            
        if merged_object.object is not None:

            if merged_object.vertex_count > 65536:
                ib_layout = buffers_format.get(f'Component{component_id}_IB', None)
                ib_buffer_semantic = ib_layout.get_element(Semantic.Index)
                ib_buffer_semantic.format = DXGIFormat.R32_UINT
                ib_buffer_semantic.stride = ib_buffer_semantic.format.byte_width * 3
                ib_layout.stride = ib_buffer_semantic.stride

            buffers, vertex_count = data_model.get_data(
                self.context, 
                self.cfg.component_collection, 
                merged_object.object, 
                merged_object.mesh, 
                self.excluded_buffers,
                buffers_format,
                self.cfg.mirror_mesh,
                index_layout)
            
            self.buffers.update(buffers)

            lod_meshes = self.extracted_object.components[component_id].lods or []
            for lod_mesh in lod_meshes:
                if len(lod_mesh.vg_map) == 0:
                    continue
                vb2 = self.buffers.get(f'Component{component_id}_VB2', None)
                if vb2 is not None:
                    indices = vb2.get_field(Semantic.Blendindices)
                    vg_count = int(indices.max()) + 1

                    remap = numpy.array([lod_mesh.vg_map.get(str(vg_id), vg_id) for vg_id in range(vg_count)])
                    
                    vb2_lod = NumpyBuffer(layout=vb2.layout, size=len(vb2.data))

                    vb2_lod.set_field(Semantic.Blendindices, remap[indices])

                    weights = vb2.get_field(Semantic.Blendweights)
                    if weights is not None:
                        vb2_lod.set_field(Semantic.Blendweights, weights)

                    self.buffers[f'Component{component_id}_VB2_LOD'] = vb2_lod

            merged_object.vertex_count = vertex_count
            merged_object.shapekeys.vertex_count = len(self.buffers.get('ShapeKeyVertexId', []))

            remapped_vgs_counts = self.buffers.pop('BlendRemapLayout', None)
            if remapped_vgs_counts is not None:
                remap_id = 0
                for component_id, vg_count in enumerate(remapped_vgs_counts.data.tolist()):
                    if vg_count == 0:
                        continue
                    component = merged_object.components[component_id]
                    if vg_count > 256:            
                        raise ConfigError('component_collection', f'Component{component_id} 256 VG limit exceeded!\n'
                                        f'Currently it consists of {len(component.objects)} object(s) using total of {vg_count} VGs with non-zero weights.\n'
                                        f'Please reduce the number of non-empty VGs or split objects between different components.')
                    component.blend_remap_id = remap_id
                    component.blend_remap_vg_count = vg_count
                    remap_id += 1
                merged_object.blend_remap_count = remap_id

        print(f'Total mesh data collection time: {time.time() - start_time :.3f}s')
    
    def build_mod_ini(self):
        start_time = time.time()

        ini_maker = IniMaker(
            cfg=self.cfg,
            scene=self.context.scene,
            mod_info=ModInfo(
                efmi_tools_version=Version(self.cfg.efmi_tools_version),
                required_efmi_version=Version(self.cfg.required_efmi_version),
                mod_name=self.cfg.mod_name,
                mod_author=self.cfg.mod_author,
                mod_desc=self.cfg.mod_desc,
                mod_link=self.cfg.mod_link,
                mod_logo=self.local_mod_logo_path,
            ),
            extracted_object=self.extracted_object,
            merged_object=self.merged_object,
            buffers=self.buffers,
            textures=self.textures,
            comment_code=self.cfg.comment_ini,
            skeleton_scale=self.cfg.skeleton_scale,
            unrestricted_custom_shape_keys=self.cfg.unrestricted_custom_shape_keys,
        )

        self.ini = ini_maker

        if self.cfg.custom_template_live_update:
            self.ini.start_live_write(self.context, self.cfg)
        else:
            self.ini.build_from_template(self.context, self.cfg, with_checksum=True)

        print(f'Total mod ini build time: {time.time() - start_time :.3f}s')

    def write_files(self):
        start_time = time.time()

        for buffer_name, buffer in self.buffers.items():
            print(f'Writing {buffer_name}.buf...')
            with open(self.meshes_path / f'{buffer_name}.buf', 'wb') as f:
                f.write(buffer.get_bytes())

        if not self.cfg.partial_export:
            # Write textures
            if self.cfg.copy_textures:
                for texture in self.textures:
                    texture_path = self.textures_path / texture.filename
                    if texture_path.is_file():
                        continue
                    print(f'Copying {texture_path.name}...')
                    shutil.copy(texture.path, texture_path)
            # Write mod logo
            mod_logo_path = resolve_path(self.cfg.mod_logo)
            if mod_logo_path.is_file():
                print(f'Copying {self.local_mod_logo_path.name}...')
                shutil.copy(mod_logo_path, self.local_mod_logo_path)
            # Write mod.ini
            if self.cfg.write_ini:
                self.ini.write(ini_path=self.mod_output_folder / 'mod.ini')
                # self.ini.write(ini_path=self.mod_output_folder / 'mod_old.ini', ini_string=self.ini.build_old())
                
        print(f'Disk write time: {time.time() - start_time :.3f}s')

    def compare_outputs(self, old_path: Path, new_path: Path):

        global data_models
        data_model = data_models['EFMI']

        for buffer_name, layout in data_model.buffers_format.items():

            print(f'Comparing {buffer_name}.buf buffers...')

            with open(old_path / (buffer_name + '.buf'), 'rb') as f1, open(new_path / (buffer_name + '.buf'), 'rb') as f2:
                
                old_buffer = NumpyBuffer(layout)
                old_buffer.import_raw_data(f1.read())

                new_buffer = NumpyBuffer(layout)
                new_buffer.import_raw_data(f2.read())

                for semantic in layout.semantics:

                    old_semantic_data = old_buffer.get_field(semantic.get_name()).tolist()
                    new_semantic_data = new_buffer.get_field(semantic.get_name()).tolist()

                    if old_semantic_data == new_semantic_data:
                        print(f'{buffer_name} {semantic.abstract} matches!')
                    else:
                        # print(f'{buffer_name} {semantic.abstract} differs:')

                        verbose = True
                        if buffer_name == 'Vector':
                            print(f'Comparing {semantic.abstract} in silent mode...')
                            verbose = False
                        else:
                            print(f'Comparing {semantic.abstract} in verbose mode...')

                        num_diffs = 0

                        for i in range(len(old_semantic_data)):
                            old_data = old_semantic_data[i]
                            new_data = new_semantic_data[i]

                            if old_data != new_data:
                                num_diffs += 1
                                if verbose:
                                    print(f'Element {i} diff: {old_data} != {new_data}')

                        print(f'Found {num_diffs} diffs (out of {len(old_semantic_data)} entries)')


def blender_export(operator, context, cfg, excluded_buffers):
    mod_exporter = ModExporter(context, cfg, excluded_buffers)
    mod_exporter.export_mod()
