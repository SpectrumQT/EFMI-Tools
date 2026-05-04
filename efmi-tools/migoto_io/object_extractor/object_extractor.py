import time

from pathlib import Path
from dataclasses import dataclass, field

from ..migoto_model.log_model.log_model import FrameDumpLog
from ..migoto_model.frame_model.frame_model import DumpModel, ParseDumpModelConfig

from .raw_object.raw_object_extractor import RawObjectExtractor, RawObjectIdentifier, DrawCallFilter, RawObjectFilter
from .migoto_object.migoto_object_builder import MigotoObjectBuilder, MigotoObject, MigotoComponent, MigotoObjectFilter
from .migoto_object.textures_descriptor import TexturesDescriptor, TextureFilter
from .migoto_object.migoto_object_exporter import ObjectExporter


@dataclass
class ObjectExtractor:
    verbose_logging: bool = False

    def build_frame_model(self, dump_path: Path) -> DumpModel:
        print(f'Processing frame dump: {dump_path}...')

        t = time.time()

        print(f'Building log model of log.txt...')

        with open(dump_path / "log.txt", 'r', encoding='utf-8') as f:
            log = FrameDumpLog.from_text(f.read(), skip_migoto_lines=True)

        print(f'Done building log model in {time.time() - t:.2f}s.')

        t = time.time()

        print(f'Building frame model...')

        dump_model_cfg = ParseDumpModelConfig(
            dump_path=dump_path,
        )
        dump_model_cfg.shader_call_config.command_config.skip_commands = {
            'Begin', 'End', 'Map', 'Unmap', 'GetData', 'GetType',
            'RSSetViewports', 'RSSetScissorRects', 'RSSetState',
            'OMGetRenderTargets', 'OMSetDepthStencilState', 'OMSetBlendState',
            'ClearRenderTargetView', 'ClearDepthStencilView', 'IASetInputLayout'
        }
        dump_model_cfg.shader_call_config.command_config.skip_stage_commands = {
            'GetSamplers', 'SetSamplers', 'GetShader'
        }

        model = DumpModel.from_frame_dump_log(log, dump_model_cfg)

        print(f'Done building frame model in {time.time() - t:.2f}s.')

        t = time.time()

        print(f'Evaluating frame model...')

        model.execute_commands()

        print(f'Done evaluating frame model in {time.time() - t:.2f}s.')

        return model

    def extract_objects(
        self,
        model: DumpModel,
        raw_object_filter: RawObjectFilter,
        migoto_object_filter: MigotoObjectFilter
    ) -> list[MigotoObject]:

        t = time.time()

        print(f'Extracting raw objects from frame model...')

        raw_objects = RawObjectExtractor(
            draw_call_filter=DrawCallFilter(),
            identifier=RawObjectIdentifier(),
            raw_object_filter=raw_object_filter,
        ).extract(model)

        print(f'Done extracting raw objects from frame model in {time.time() - t:.2f}s.')

        t = time.time()

        print(f'Building export objects...')

        migoto_object_builder = MigotoObjectBuilder(
            migoto_object_filter=migoto_object_filter,
            verbose_logging=self.verbose_logging
        )

        migoto_objects = migoto_object_builder.build(raw_objects)

        print(f'Done building {len(migoto_objects)} export objects in {time.time() - t:.2f}s.')

        return migoto_objects

    def export_objects(self, migoto_objects: list[MigotoObject], texture_filter: TextureFilter, output_path: Path):
        t = time.time()

        print(f'Exporting objects...')

        # output_path = Path(r"C:\Games\XXMI Launcher\Importers\EFMI\EFMI_DEV\Extracted Objects")

        object_exporter = ObjectExporter()

        for migoto_object in migoto_objects:
            print(f"Writing {migoto_object.id}...")
            object_output_path = output_path / migoto_object.id

            textures_descriptor = TexturesDescriptor.from_migoto_object(migoto_object, texture_filter)
            object_exporter.export(object_output_path, migoto_object, textures_descriptor)

        print(f'Done exporting objects in {time.time() - t:.2f}s.')
