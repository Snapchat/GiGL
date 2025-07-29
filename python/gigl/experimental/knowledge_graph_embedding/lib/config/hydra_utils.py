from enum import Enum
from typing import Any, Type, Union


def get_fully_qualified_name(cls_or_obj: Union[Type, Any]) -> str:
    """
    Returns the fully qualified name (module.ClassName) for a class or object.
    Handles both classes and instances by getting the type of the instance.
    """
    cls = cls_or_obj if isinstance(cls_or_obj, type) else type(cls_or_obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def build_hydra_dict_from_object(obj: Any) -> Any:
    """
    Builds a dictionary from a dataclass or namedtuple,
    recursively processing its attributes and adding '_target_'.
    Autonomously converts NewType instances to their base string.
    """
    # 1. Handle NamedTuples
    if isinstance(obj, tuple) and hasattr(obj, "_asdict"):
        data_dict = {}
        obj_type = type(obj)
        target_path = get_fully_qualified_name(obj_type)
        if target_path:
            data_dict["_target_"] = target_path

        for field_name, field_value in obj._asdict().items():
            data_dict[field_name] = build_hydra_dict_from_object(field_value)
        return data_dict

    # 2. Handle Dataclasses
    elif hasattr(obj, "__dataclass_fields__"):
        data_dict = {}
        obj_type = type(obj)
        target_path = get_fully_qualified_name(obj_type)
        if target_path:
            data_dict["_target_"] = target_path

        for field in obj.__dataclass_fields__.values():
            field_value = getattr(obj, field.name)
            data_dict[field.name] = build_hydra_dict_from_object(field_value)
        return data_dict

    # 3. Handle Lists (recursive)
    elif isinstance(obj, list):
        return [build_hydra_dict_from_object(item) for item in obj]

    # 4. Handle Enum instances generically
    elif isinstance(obj, Enum):
        # Convert Enum to its value. This is generic for all Enums.
        return obj.value

    # 5. Handle all other primitive types (str, int, float, bool, None)
    else:
        return obj
