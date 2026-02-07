

class TextFormatter:
    
    @staticmethod
    def extract_name_dupe_id(name):
        parts = name.split('.')
        if len(parts) == 1 or not parts[-1].isdigit():
            return 0, name
        return int(parts[-1]), '.'.join(parts[:-1])
    
    def dedupe_name(self, name, name_list):
        if name not in name_list:
            return name
        dupe_id, original_name = self.extract_name_dupe_id(name)
        for i in range(1, 999-dupe_id):
            new_name = f'{original_name}.{i:03d}'
            if new_name not in name_list:
                return new_name
        return self.dedupe_name(f'{name}.001', name_list)
    
    @staticmethod
    def extract_name_parts(name):
        if not isinstance(name, str):
            if hasattr(name, 'name'):
                name = name.name
            else:
                name = str(name)
        name = name.replace('$', '').replace('-', ' ').replace('.', ' ').replace('_', ' ')
        parts = list(map(str.lower, map(str.strip, name.split(' '))))
        return parts

    def format_name_camel_case(self, name):
        parts = self.extract_name_parts(name)
        return ''.join(map(str.capitalize, parts))

    def format_ini_swapvar(self, name):
        parts = self.extract_name_parts(name)
        return f"$swapvar_{'_'.join([x for x in parts if x and x not in ['var', 'swap']])}"
    
    def format_ini_drawvar(self, name):
        parts = self.extract_name_parts(name)
        return f"$draw_{'_'.join([x for x in parts if x])}"

    def extract_hotkeys_parts(self, hotkeys):
        hotkeys = hotkeys.upper().replace(',', ' ').replace('+', ' ').replace(';', ' ').replace('-', ' ')
        parts = [x for x in map(str.upper, map(str.strip, hotkeys.split(' '))) if x]
        return parts

    def format_hotkeys(self, hotkeys, join_arg=' '):
        return [join_arg.join(self.extract_hotkeys_parts(binding)) for binding in hotkeys.split(';')]
