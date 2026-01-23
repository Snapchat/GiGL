from google.protobuf import message


def assert_proto_has_field(proto: message.Message, field_name: str) -> None:
    """
    Assert that a proto has a defined field. Throws an AssertionError if the field is not set.
    Note: this method only works for message fields, not for singular (non-message) fields.

    Args:
        proto: A proto message.
        field_name: A string representing the field name.
    """

    assert proto.HasField(
        field_name
    ), f"Invalid '{field_name}'; must provide {field_name}."


def assert_proto_field_value_is_truthy(proto: message.Message, field_name: str) -> None:
    """
    Assert that a proto field is not empty. Throws an AssertionError if the field is not set or if the value in the field is equal to the default value.

    Args:
        proto: A proto message.
        field_name: A string representing the field name.
    """

    assert getattr(proto, field_name), (
        f"Invalid '{field_name}'; value must be "
        "positive (if integer) or non-empty (if string)."
    )
