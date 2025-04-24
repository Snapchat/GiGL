from gigl.common import GcsUri, Uri, UriFactory


def get_gcs_assets_dir_from_frozen_gbml_config_uri(
    gbml_config_uri: Uri,
) -> GcsUri:
    # hacky; we currently don't have a way to get gcs assets dir from the gbml config directly
    gcs_tokens = gbml_config_uri.uri.split("/")
    uri = UriFactory.create_uri(uri="/".join(gcs_tokens[:-1]))
    assert isinstance(uri, GcsUri), f"Expected URI of type {GcsUri.__name__}"
    return uri
