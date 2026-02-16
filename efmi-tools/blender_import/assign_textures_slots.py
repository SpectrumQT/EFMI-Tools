import bpy
import re
import json
from pathlib import Path

BC7_UNORM_SRGB = 0x63


def dds_format_byte(filepath):
    try:
        with open(filepath, "rb") as f:
            f.seek(0x80)
            return f.read(1)[0]
    except Exception:
        return None


def parse_mesh(mesh_name):
    match = re.match(r'Component\s+(\d+)\s+([a-f0-9]+)', mesh_name, re.IGNORECASE)
    if not match:
        return None
    return {'component_id': match.group(1), 'hash': match.group(2)}


class TextureAssigner:
    @staticmethod
    def run(texture_folder, debug=False):
        folder = Path(texture_folder)
        if not folder.exists():
            print(f"Texture folder not found: {folder}")
            return

        comp_to_data = TextureAssigner._load_texture_usage(folder)
        if not comp_to_data:
            print("No mappings loaded, aborting.")
            return

        hash_to_file = TextureAssigner._build_hash_to_filepath(folder)

        meshes_to_process = [obj for obj in bpy.data.objects
                             if obj.type == 'MESH' and not obj.data.materials]

        if not meshes_to_process:
            print("No meshes without materials found, nothing to assign.")
            return

        assigned     = []
        missing_tex  = []
        missing_json = []

        for obj in meshes_to_process:
            info = parse_mesh(obj.name)
            if not info:
                continue

            comp_data = comp_to_data.get(info['component_id'])
            if comp_data is None:
                missing_json.append(obj.name)
                continue

            filepath = TextureAssigner._resolve_texture(info['component_id'], comp_data, hash_to_file, debug)
            if filepath is None:
                missing_tex.append((obj.name, comp_data['t0']))
                continue

            TextureAssigner._assign_image_to_mesh(obj, filepath)
            print(f"  {obj.name}  ←  {filepath.name}")
            assigned.append(obj.name)

        print(f"\nAssigned: {len(assigned)} | Missing JSON: {len(missing_json)} | Missing DDS: {len(missing_tex)}")

        if missing_json:
            for m in missing_json:
                print(f"  [no json]  {m}")

        if missing_tex:
            for mesh, hashes in missing_tex:
                print(f"  [no dds]   {mesh}  ({', '.join(hashes)})")

    @staticmethod
    def _load_texture_usage(folder):
        json_path = folder / "TextureUsage.json"
        if not json_path.exists():
            print(f"TextureUsage.json not found in: {folder}")
            return {}

        data = json.loads(json_path.read_text(encoding="utf-8"))
        mapping = {}

        for component_name, slots in data.items():
            id_match = re.search(r'\d+', component_name)
            if not id_match:
                continue

            t0_entries = slots.get("ps-t0", [])
            if not t0_entries:
                continue

            t0_hashes = []
            for entry in t0_entries:
                h = re.match(r'^([a-f0-9]+)-vs=', entry, re.IGNORECASE)
                if h:
                    t0_hashes.append(h.group(1).lower())

            if not t0_hashes:
                continue

            t10_t19_counts = {}
            for slot_name, entries in slots.items():
                if not re.match(r'^ps-t1[0-9]$', slot_name):
                    continue
                for entry in entries:
                    h = re.match(r'^([a-f0-9]+)-vs=', entry, re.IGNORECASE)
                    if h:
                        key = h.group(1).lower()
                        t10_t19_counts[key] = t10_t19_counts.get(key, 0) + 1

            mapping[id_match.group(0)] = {
                't0': t0_hashes,
                't10_t19_counts': t10_t19_counts,
            }

        return mapping

    @staticmethod
    def _build_hash_to_filepath(folder):
        return {
            m.group(0).lower(): path
            for path in folder.glob("*.dds")
            if (m := re.search(r'[a-f0-9]{8}', path.name, re.IGNORECASE))
        }

    @staticmethod
    def _resolve_texture(comp_id, comp_data, hash_to_file, debug):
        seen = set()
        candidates = []
        for h in comp_data['t0']:
            if h not in seen and (path := hash_to_file.get(h)):
                seen.add(h)
                candidates.append((h, path))

        if not candidates:
            return None

        if len(candidates) == 1:
            if debug:
                print(f"  [Component {comp_id}] {candidates[0][1].name}")
            return candidates[0][1]

        t10_t19_counts = comp_data['t10_t19_counts']
        scored = sorted(candidates, key=lambda x: t10_t19_counts.get(x[0], 0), reverse=True)
        top_count = t10_t19_counts.get(scored[0][0], 0)
        tied = [(h, path) for h, path in scored if t10_t19_counts.get(h, 0) == top_count]

        if len(tied) == 1:
            if debug:
                print(f"  [Component {comp_id}] {tied[0][1].name}")
            return tied[0][1]

        srgb = [(h, path) for h, path in tied if dds_format_byte(path) == BC7_UNORM_SRGB]

        if len(srgb) == 1:
            if debug:
                print(f"  [Component {comp_id}] {srgb[0][1].name}")
            return srgb[0][1]

        pool = srgb if srgb else tied
        result = max(pool, key=lambda x: x[1].stat().st_size)[1]
        if debug:
            print(f"  [Component {comp_id}] {result.name}")
        return result

    @staticmethod
    def _assign_image_to_mesh(obj, filepath):
        mat = (bpy.data.materials.get(f"Mat_{obj.name}") or
               bpy.data.materials.new(name=f"Mat_{obj.name}"))
        mat.use_nodes = True

        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        node_tex  = nodes.new('ShaderNodeTexImage')
        node_bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        node_out  = nodes.new('ShaderNodeOutputMaterial')

        node_tex.location  = (-300, 0)
        node_bsdf.location = (0, 0)
        node_out.location  = (300, 0)

        img = (bpy.data.images.get(filepath.name) or
               bpy.data.images.load(str(filepath)))
        img.colorspace_settings.name = 'sRGB'
        node_tex.image = img
        img.alpha_mode = 'NONE'

        links.new(node_tex.outputs['Color'], node_bsdf.inputs['Base Color'])
        links.new(node_bsdf.outputs['BSDF'], node_out.inputs['Surface'])


def assign_textures(texture_folder, debug=False):
    TextureAssigner.run(texture_folder, debug)