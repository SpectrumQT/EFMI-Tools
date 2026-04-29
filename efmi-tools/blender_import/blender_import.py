import time

from ..addon.exceptions import ConfigError

from ..migoto_io.blender_interface.utility import *
from ..migoto_io.blender_interface.collections import *
from ..migoto_io.blender_interface.objects import *

from ..migoto_io.object_extractor.migoto_object.migoto_object import MigotoObject, MigotoComponent

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
        # if cfg.import_skeleton_type == 'MERGED':
        #     component_pattern = re.compile(r'.*component[ -_]*([0-9]+).*')
        #     result = component_pattern.findall(fmt_path.name.lower())
        #     if len(result) == 1:
        #         component = extracted_object.components[int(result[0])]
        #         vg_remap = numpy.array(list(component.vg_map.values()))

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

        # if cfg.skip_empty_vertex_groups and cfg.import_skeleton_type == 'MERGED':
        #     remove_unused_vertex_groups(context, obj)

        num_shapekeys = 0 if obj.data.shape_keys is None else len(getattr(obj.data.shape_keys, 'key_blocks', []))

        print(f'{component.metadata.mesh_name} import time: {time.time()-start_time :.3f}s ({len(obj.data.vertices)} vertices, {len(obj.data.loops)} indices, {num_shapekeys} shapekeys)')

    col = new_collection(collection_name)
    for obj in imported_objects:
        link_object_to_collection(obj, col)


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
