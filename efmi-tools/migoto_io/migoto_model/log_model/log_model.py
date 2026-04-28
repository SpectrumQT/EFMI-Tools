from dataclasses import dataclass


@dataclass
class DumpedBinding:
    line_id: int
    line: str

    @classmethod
    def from_line(cls, line_id: int, line: str) -> "DumpedBinding":
        return cls(line_id=line_id, line=line.strip())


@dataclass
class DumpedCommand:
    line_id: int
    line: str
    bindings: list[DumpedBinding]

    @classmethod
    def from_block(
        cls,
        command_line_id: int,
        command_line: str,
        binding_lines: list[tuple[int, str]],
    ) -> "DumpedCommand":
        bindings = [
            DumpedBinding.from_line(line_id, line)
            for line_id, line in binding_lines
        ]
        return cls(
            line_id=command_line_id,
            line=command_line.strip(),
            bindings=bindings,
        )


@dataclass
class DumpedCall:
    id: int
    commands: list[DumpedCommand]

    @classmethod
    def from_block(
        cls,
        id: int,
        command_blocks: list[tuple[int, str, list[tuple[int, str]]]]
    ) -> "DumpedCall":
        commands = [
            DumpedCommand.from_block(line_id, line, bindings)
            for line_id, line, bindings in command_blocks
        ]
        return cls(id=id, commands=commands)


ignore = ['GetData', 'Map', 'Unmap', 'Begin', 'End', 'RSSetViewports', 'RSSetScissorRects', 'ClearDepthStencilView',
          'OMSetDepthStencilState', 'OMSetBlendState', 'IASetInputLayout', 'PSSetSamplers', 'ClearRenderTargetView',
          'CSSetSamplers', 'RSSetState', 'VSSetSamplers', 'CSGetShader', 'CSGetSamplers', 'OMGetRenderTargets', 'GetType']




@dataclass
class FrameDumpLog:
    calls: list[DumpedCall]
    parse_errors: list[str]

    @classmethod
    def from_text(
        cls,
        text: str,
        skip_migoto_lines: bool = False,
        strict_mode: bool = False,
        ignore_commands: list[str] | None = None,
    ) -> "FrameDumpLog":
        """Parse a FrameDumpLog from a raw text string."""
        lines = text.splitlines()
        if not lines:
            raise ValueError(f'Cannot create FrameDumpLog from empty text!')

        # Skip analyse_options header (always first line)
        start_index = 1 if lines[0].startswith("analyse_options:") else 0

        calls: list[DumpedCall] = []
        parse_errors: list[str] = []

        current_call_id: int | None = None
        current_command_blocks: list[tuple[int, str, list[tuple[int, str]]]] = []

        current_command_line_id: int | None = None
        current_command_line: str | None = None
        current_binding_lines: list[tuple[int, str]] = []

        def get_call_id(line: str) -> int:
            if len(line) >= 7 and line[6] == " ":
                prefix = line[:6]
                if prefix.isdigit():
                    return int(prefix)
            return -1

        def flush_command():
            nonlocal current_command_line_id, current_command_line, current_binding_lines, current_command_blocks
            if current_command_line is not None:
                block = (current_command_line_id, current_command_line, current_binding_lines)
                current_command_blocks.append(block)
            current_command_line_id = None
            current_command_line = None
            current_binding_lines = []

        def flush_call():
            nonlocal current_command_blocks, calls
            if current_command_blocks:
                calls.append(DumpedCall.from_block(current_call_id, current_command_blocks))
            current_command_blocks = []

        def parse_line(line_id: int, raw_line: str):
            nonlocal current_call_id, current_command_line_id, current_command_line

            line = raw_line.rstrip()
            if not line:
                return

            line_text = line[7:]
            if skip_migoto_lines and line_text.startswith("3DMigoto") and not line_text.startswith("3DMigoto Dumping"):
                return

            call_id = get_call_id(line)

            # New binding line for current command
            if call_id == -1:
                if line[0].strip():
                    raise ValueError(f'non-empty line prefix `{line[0]}` for non-command line')
                if current_command_line is None:
                    raise ValueError(f'binding line found without parent command line')
                current_binding_lines.append((line_id, line))
                return

            # New call
            if call_id != current_call_id:
                flush_command()
                flush_call()
                current_call_id = call_id

            # New command
            flush_command()
            current_command_line_id = line_id
            current_command_line = line_text

        for line_id in range(start_index, len(lines)):
            raw_line = lines[line_id]
            try:
                parse_line(line_id, raw_line)
            except Exception as e:
                if strict_mode:
                    raise ValueError(f'{str(e)} for line {line_id} `{raw_line}`') from e
                else:
                    parse_errors.append(f'[{line_id}][Error]: {str(e)} for line `{raw_line}`')

        # Final flush
        flush_command()
        flush_call()

        return cls(calls=calls, parse_errors=parse_errors)
