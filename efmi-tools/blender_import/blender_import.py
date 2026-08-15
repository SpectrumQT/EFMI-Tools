import time

from textwrap import dedent

from ..addon.exceptions import ConfigError

from ..migoto_io.blender_interface.utility import *
from ..migoto_io.blender_interface.collections import *
from ..migoto_io.blender_interface.objects import *

from ..migoto_io.object_extractor.migoto_object.migoto_object import MigotoObject, MigotoComponent
from ..migoto_io.blender_tools.vertex_groups import remove_unused_vertex_groups
from ..migoto_io.migoto_model.migoto_mesh import WeightingType

from ..data_models.data_model_efmi import DataModelEFMI


# TODO: Add support of import of unhandled semantics into vertex attributes

def import_object(
    context,
    cfg,
    collection_name: str,
    migoto_object: MigotoObject,
    extended_mesh_name: bool = False,
):
    model = DataModelEFMI()
    model.legacy_vertex_colors = cfg.color_storage == 'LEGACY'

    if migoto_object.metadata.format_version < 4 and cfg.import_skeleton_type == 'MERGED':
        raise ConfigError('object_source_folder', f"""
            Specified sources folder uses old data format `v{migoto_object.metadata.format_version}`!
            This format is missing data required for Merged Skeleton.
            Please extract object again from a new frame dump.
        """)

    if migoto_object.metadata.weigthing_type != WeightingType.Explicit:
        raise ConfigError('import_skeleton_type', f"""
            Specified sources folder contains object {'without' if migoto_object.metadata.weigthing_type == WeightingType.NoWeights else 'with implicit'} weights!
            Merged Skeleton makes sense only for object with explicit weights.
            Please use Per-Component Skeleton instead.
        """)

    if migoto_object.metadata.format_version < 3:
        cfg.last_error_setting_name = "object_source_folder"
        cfg.last_error_text = dedent(f"""
            Specified sources folder uses outdated data format `v{migoto_object.metadata.format_version}`!
            When used for mod export, it will not work correctly.
            Please extract object again from a new frame dump.
        """).strip()

    imported_objects = []

    for component in migoto_object.components:
        start_time = time.time()

        if extended_mesh_name:
            mesh_name = f"{component.metadata.mesh_name} {component.metadata.ib_hash}"
            if component.metadata.cpu_posed:
                mesh_name += f" CPU-posed (only textures modding supported)"
        else:
            mesh_name = component.metadata.mesh_name

        mesh = bpy.data.meshes.new(mesh_name)
        obj = bpy.data.objects.new(mesh.name, mesh)

        vg_remap = None
        if cfg.import_skeleton_type == 'MERGED' and not component.metadata.cpu_posed:
            if not component.metadata.vg_map:
                raise ConfigError('object_source_folder', f"""
                    Specified sources folder contains object with invalid data!
                    {component.metadata.mesh_name} is missing `vg_map` required for import in Merged Skeleton mode.
                    Most likely this object is currently incompatible with Merged Skeleton.
                    Please use Per-Component Skeleton instead.
                """)
            vg_remap = numpy.array(list(component.metadata.vg_map.values()))

        model.set_data(
            obj=obj,
            mesh=mesh,
            index_buffer=component.mesh.index_buffer,
            vertex_buffer=component.mesh.vertex_buffer,
            vg_remap=vg_remap,
            mirror_mesh=cfg.mirror_mesh,
            mesh_scale=1.00,
            mesh_rotation=migoto_object.metadata.rotation.to_tuple(),
            import_tangent_data_to_attribute=cfg.import_tangent_data_to_attribute,
        )
        imported_objects.append(obj)

        num_shapekeys = 0 if obj.data.shape_keys is None else len(getattr(obj.data.shape_keys, 'key_blocks', []))

        print(f'{component.metadata.mesh_name} import time: {time.time()-start_time :.3f}s ({len(obj.data.vertices)} vertices, {len(obj.data.loops)} indices, {num_shapekeys} shapekeys)')

    col = new_collection(collection_name)
    for obj in imported_objects:
        link_object_to_collection(obj, col)
        if cfg.skip_empty_vertex_groups and cfg.import_skeleton_type == 'MERGED':
            remove_unused_vertex_groups(context, obj)


def blender_import(operator, context, cfg):
    start_time = time.time()

    object_source_folder = resolve_path(cfg.object_source_folder)

    print(f"Object import started for '{object_source_folder.stem}' folder")

    if not object_source_folder.is_dir():
        raise ConfigError('object_source_folder', 'Specified sources folder does not exist!')

    metadata_path = object_source_folder / 'Metadata.json'
    if not metadata_path.is_file():
        raise ConfigError('object_source_folder', 'Specified folder is missing Metadata.json!')

    try:
        migoto_object = MigotoObject.from_exported_files(object_source_folder, metadata_path)
    except Exception as e:
        raise ConfigError('object_source_folder', f'Failed to load object from sources folder:\n{e}')

    collection_name = object_source_folder.stem

    try:
        import_object(context, cfg, collection_name, migoto_object, extended_mesh_name=True)
    except Exception as e:
        raise ConfigError('object_source_folder', f'Failed to import object from sources folder:\n{e}')

    print(f'Total import time: {time.time() - start_time :.3f}s')
