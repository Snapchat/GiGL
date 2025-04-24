import unittest

from gigl.src.inference.v1.lib.inference_output_schema import (
    InferenceOutputBigqueryTableSchema,
    InferenceOutputBigqueryTableSchemaBuilder,
)


class InferenceOutputSchemaBuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.field_one = "one"
        self.field_two = "two"
        self.field_type = "INTEGER"
        self.field_mode = "NULLABLE"

    def __check_builder_is_empty(
        self, builder: InferenceOutputBigqueryTableSchemaBuilder
    ):
        self.assertDictEqual(builder._fields, {})
        self.assertIsNone(builder._node_field)

    def test_builder_can_build(self):
        builder: InferenceOutputBigqueryTableSchemaBuilder = (
            InferenceOutputBigqueryTableSchemaBuilder()
        )

        # Check the builder is empty.
        self.__check_builder_is_empty(builder=builder)

        expected_schema_dict = {
            "fields": [
                {
                    "name": self.field_one,
                    "type": self.field_type,
                    "mode": self.field_mode,
                },
                {
                    "name": self.field_two,
                    "type": self.field_type,
                    "mode": self.field_mode,
                },
            ]
        }
        expected_schema = InferenceOutputBigqueryTableSchema(
            schema=expected_schema_dict,
            node_field=self.field_one,
        )

        # Add to builder.
        builder.add_field(
            name=self.field_one, field_type=self.field_type, mode=self.field_mode
        )
        builder.add_field(
            name=self.field_two, field_type=self.field_type, mode=self.field_mode
        )
        builder.register_node_field(name=self.field_one)

        # Ensure builder fields are not empty.
        self.assertIsNotNone(builder._fields)
        self.assertIsNotNone(builder._node_field)

        # Build DatasetSamples object.
        schema = builder.build()

        # Check the builder is now empty and the object is successfully built.
        self.__check_builder_is_empty(builder=builder)
        self.assertEqual(schema, expected_schema)

    def test_builder_fails_without_required_fields(self):
        builder: InferenceOutputBigqueryTableSchemaBuilder = (
            InferenceOutputBigqueryTableSchemaBuilder()
        )

        builder.add_field(
            name=self.field_one, field_type=self.field_type, mode=self.field_mode
        )
        # Should fail without a registered node field.
        with self.assertRaises(AssertionError):
            builder.build()

        builder.reset()

        # Should fail when trying to register an unseen node field
        with self.assertRaises(AssertionError):
            builder.register_node_field(name=self.field_one)
