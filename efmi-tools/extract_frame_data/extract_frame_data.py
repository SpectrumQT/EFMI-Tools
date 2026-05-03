import time

from textwrap import dedent

from ..addon.exceptions import ConfigError

from ..migoto_io.blender_interface.utility import *
from ..migoto_io.blender_interface.collections import *
from ..migoto_io.blender_interface.objects import *

from ..migoto_io.migoto_model.migoto_mesh import GeometryMatcherConfig, GeometryMatcherMethod

from ..migoto_io.object_extractor.raw_object.raw_object_extractor import RawObjectFilter
from ..migoto_io.object_extractor.migoto_object.migoto_object import MigotoObject, MigotoComponent, DuplicateDataError
from ..migoto_io.object_extractor.migoto_object.migoto_object_builder import MigotoObject, MigotoObjectFilter
from ..migoto_io.object_extractor.migoto_object.textures_descriptor import TextureFilter
from ..migoto_io.object_extractor.object_extractor import ObjectExtractor
from ..migoto_io.object_extractor.lod_matcher import LODMatcher, ObjectLowSimilarityError, ComponentLowSimilarityError

from ..blender_import.blender_import import import_object


def match_lods(
    cfg,
    full_object: MigotoObject,
    lod_candidate_objects: list[MigotoObject],
 ) -> tuple[MigotoObject, dict[MigotoComponent, tuple[MigotoComponent, dict[int, int] | None]]]:
    error_threshold = cfg.geo_matcher_voxel_error_threshold if cfg.geo_matcher_method == 'VOXEL' else cfg.geo_matcher_error_threshold
    lod_matcher = LODMatcher(
        component_min_vertex_count=cfg.skip_component_below_vertex_count if cfg.skip_component_below_vertex_count_enabled else 0,
        component_hash_blacklist=cfg.skip_component_hashes if cfg.skip_component_hashes_enabled else "",
        object_similarity_threshold=error_threshold,
        component_similarity_threshold=error_threshold,
        geo_matcher_main_config=GeometryMatcherConfig(
            method=GeometryMatcherMethod(cfg.geo_matcher_method),
            sensitivity=cfg.geo_matcher_sensivity,
            voxel_size=cfg.geo_matcher_voxel_size,
            samples_count=cfg.geo_matcher_sample_size,
        ),
        geo_matcher_prefilter_config=GeometryMatcherConfig(
            method=GeometryMatcherMethod(cfg.geo_matcher_method),
            sensitivity=cfg.geo_matcher_sensivity,
            voxel_size=cfg.geo_matcher_prefilter_voxel_size,
            samples_count=cfg.geo_matcher_prefilter_sample_size,
        ),
        geo_matcher_prefilter_candidates_count=cfg.geo_matcher_prefilter_candidates_count,
        vg_matcher_candidates_count=cfg.vg_matcher_candidates_count,
    )

    lod_object, matched_components = lod_matcher.find_matching_lods(full_object, lod_candidate_objects)

    return lod_object, matched_components


def import_lods(
    context,
    cfg,
    full_object: MigotoObject,
    lod_object: MigotoObject,
    matched_components: dict[MigotoComponent, tuple[MigotoComponent, dict[int, int] | None]],
):

    for full_component in full_object.components:
        (lod_component, vg_map) = matched_components.get(full_component, (None, None))
        if lod_component is None:
            lod_component = full_component
        try:
            full_component.import_lod_metadata(lod_object.id, lod_component, vg_map, cfg.allow_lod_overwrite)
        except DuplicateDataError as e:
            raise ConfigError('allow_lod_overwrite', dedent(f"""
                {e}
                To forcefully overwrite it, retry LoD import with "Allow LoD Data Overwrite" enabled
            """))

    object_source_folder = resolve_path(cfg.object_source_folder)
    full_object.export_metadata(object_source_folder)

    # Import lod mesh for debug
    if cfg.import_matched_lod_objects:

        lod_level = max([lod_id for lod_id, lod in enumerate(full_object.metadata.components[0].lods)]) + 1
        collection_name = f"{full_object.id} LOD{lod_level}: {lod_object.id}"

        for full_component in full_object.components:
            (lod_component, _) = matched_components.get(full_component, (None, None))
            if lod_component is None:
                continue
            mesh_name = f"{full_component.metadata.mesh_name} full={full_component.metadata.ib_hash} lod={lod_component.metadata.ib_hash}"
            if lod_component.metadata.ib_hash == full_component.metadata.ib_hash:
                if lod_component.metadata.vg_map:
                    mesh_name += f" (full mesh, full skeleton)"
                else:
                    mesh_name += f" (full mesh, simplified skeleton)"
            else:
                mesh_name += f" (simplified mesh and skeleton)"
            lod_component.metadata.mesh_name = mesh_name

        print("Non-matched components:")
        matched_lod_components = [lod_component for lod_component, _ in matched_components.values() if lod_component]
        for lod_component in lod_object.components:
            if lod_component.metadata.mesh_name.startswith("Skipped"):
                print(lod_component.metadata.mesh_name)
                continue
            if lod_component not in matched_lod_components:
                lod_component.metadata.mesh_name = f"Skipped Component ib={lod_component.metadata.ib_hash} (no matching full component found)"
                print(lod_component.metadata.mesh_name)

        print(f"Importing object {lod_object.id} to Blender...")
        import_object(context, cfg, collection_name, lod_object)


def extract_frame_data(context, cfg, extract_lods=False):
    if not extract_lods:
        dump_path = resolve_path(cfg.frame_dump_folder)
    else:
        dump_path = resolve_path(cfg.lod_frame_dump_folder)

    if not dump_path.is_dir():
        raise ConfigError('frame_dump_folder', 'Specified dump folder does not exist!')
    if not Path(dump_path / 'log.txt').is_file():
        raise ConfigError('frame_dump_folder', 'Specified dump folder is missing log.txt file!')

    start_time = time.time()

    object_extractor = ObjectExtractor(
        verbose_logging=cfg.verbose_logging,
    )

    frame_model = object_extractor.build_frame_model(dump_path)

    migoto_objects = object_extractor.extract_objects(
        model=frame_model,
        raw_object_filter=RawObjectFilter(
            min_component_count=cfg.skip_object_min_component_count if cfg.skip_object_min_component_count_enabled else 0,
            min_texture_count=cfg.skip_object_min_texture_count if cfg.skip_object_min_texture_count_enabled else 0,
            lookup_resource_hashes=cfg.skip_object_resource_hashes if cfg.skip_object_resource_hashes_enabled else "",
        ),
        migoto_object_filter=MigotoObjectFilter(
            ignore_errors=cfg.tolerate_extraction_errors,
            skip_static_objects=cfg.skip_static_objects,
        ),
    )

    if not extract_lods:

        texture_filter = TextureFilter(
            exclude_extensions=["jpg", "buf"] if cfg.skip_jpg_textures else ["buf"],
            exclude_hashes=[],
            min_file_size=cfg.skip_small_textures_size * 1024 if cfg.skip_small_textures else 0,
        )

        output_path = resolve_path(cfg.extract_output_folder)

        object_extractor.export_objects(migoto_objects, texture_filter, output_path)

        if cfg.import_extracted_objects:
            for migoto_object in migoto_objects:
                import_object(context, cfg, migoto_object.id, migoto_object, extended_mesh_name=True)

    else:

        object_source_folder = resolve_path(cfg.object_source_folder)

        full_object = MigotoObject.from_exported_files(object_source_folder)

        try:
            lod_object, matched_components = match_lods(cfg, full_object, migoto_objects)
        except (
            ObjectLowSimilarityError,
            ComponentLowSimilarityError,
        ) as e:
            raise ConfigError('geo_matcher_error_threshold', dedent(f"""
                {e}
                It is below configured {cfg.skip_lods_below_error_threshold:.2f}% Geometry Matcher Error Threshold.
                If it's not too far off, try to lower threshold. Otherwise either dump is missing some data or search engine fails to handle it.
            """))

        import_lods(context, cfg, full_object, lod_object, matched_components)

        lower_poly_lod_count = sum(1 for k, v in matched_components.items() if k.metadata.ib_hash != v[0].metadata.ib_hash)

        bpy.context.window_manager.popup_menu(
            lambda self, context: self.layout.label(
                text=(
                    f"Successfully imported LOD data for {len(matched_components)} components to Metadata.json"
                    f" (where {len(matched_components) - lower_poly_lod_count} components seem to use full mesh as LOD)."
                )
            ),
            title="LOD Import Complete",
            icon="INFO"
        )

    print(f"Execution time: %s seconds" % (time.time() - start_time))
