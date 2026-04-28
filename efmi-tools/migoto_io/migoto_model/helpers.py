import types
import inspect

from dataclasses import dataclass, fields
from typing import ClassVar, Any, Union
from typing import get_origin, get_args


class AutoArgsMixin:
    _arg_map_cache: ClassVar[dict | None]  = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._arg_map_cache = None

    @staticmethod
    def _identity(x):
        return x

    @classmethod
    def arg_map(cls):
        """
        Build or return a cached mapping of dataclass fields to argument names and converters.

        Returns a dictionary where:
            key: field name in the dataclass
            value: tuple (arg_name, converter)
                - arg_name: the string used as key in the input arguments dict
                - converter: a callable that converts the string value into the expected type

        Only fields with 'arg' metadata are included.
        The mapping is cached per subclass for efficiency.

        Example:
            >>> class MyCall(AutoArgsMixin):
            ...     foo: int = field(metadata={"arg": "foo"})
            >>> MyCall.arg_map()
            {'foo': ('foo', <class 'int'>)}
        """
        if cls._arg_map_cache is None:
            mapping = {}

            for f in fields(cls):
                arg = f.metadata.get("arg")
                if not arg:
                    continue

                field_type = f.type
                allow_none = False

                origin = get_origin(field_type)
                args = get_args(field_type)

                # Handle `Type | None` or `Optional[Type]`
                if origin in (Union, types.UnionType):
                    non_none = [a for a in args if a is not type(None)]

                    if len(non_none) == 1:
                        converter = non_none[0]
                        allow_none = len(args) != len(non_none)
                    else:
                        converter = cls._identity
                else:
                    converter = field_type

                if not callable(converter):
                    converter = cls._identity

                mapping[f.name] = (arg, converter, allow_none)

            cls._arg_map_cache = mapping

        return cls._arg_map_cache

    @classmethod
    def field_kwargs_from_dict(cls, arguments: dict[str, str]) -> dict[str, Any]:
        """
        Convert a dictionary of string arguments into keyword arguments for the dataclass.

        This uses the mapping from `arg_map` to:
            - select only fields with 'arg' metadata
            - convert string values to the correct type using the associated converter
            - set missing or None values to None for optional fields

        Returns a dictionary suitable to pass directly to the dataclass constructor.

        Example:
            >>> class MyCall(AutoArgsMixin):
            ...     foo: int = field(metadata={"arg": "foo"})
            ...     bar: MyEnum | None = field(metadata={"arg": "bar"})
            >>> args = {"foo": "123", "bar": None}
            >>> MyCall.field_kwargs_from_dict(args)
            {'foo': 123, 'bar': None}
        """
        kwargs = {}
        for field_name, (arg_name, converter, allow_none) in cls.arg_map().items():
            value = arguments.get(arg_name, None)
            kwargs[field_name] = converter(value) if value is not None or not allow_none else None
        return kwargs

    @staticmethod
    def parse_kv_pairs(line: str, pair_separator: str = " ", value_separator: str = "=") -> dict[str, str]:
        arguments = {}

        for pair in line.split(pair_separator):
            parts = pair.split(value_separator)
            if len(parts) != 2:
                continue

            arguments[parts[0].strip()] = parts[1].strip()

        return arguments


def raise_with_args(msg, e):
    frame = inspect.currentframe().f_back
    arg_info = inspect.getargvalues(frame)
    args = {k: arg_info.locals[k] for k in arg_info.args}
    args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
    # import traceback
    # traceback.print_exc()
    raise ValueError(f"{msg}: {args_str}") from e
