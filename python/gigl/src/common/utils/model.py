import tempfile
from typing import OrderedDict

import torch

from gigl.common import LocalUri, Uri
from gigl.src.common.utils.file_loader import FileLoader


def save_scripted_model(model: torch.nn.Module, save_to_path_uri: Uri) -> None:
    assert isinstance(
        model, torch.nn.Module
    ), "Can only save model of type torch.nn.Module"
    file_loader = FileLoader()
    tmp_save_model_file = tempfile.NamedTemporaryFile(delete=False)

    model_scripted = torch.jit.script(model)  # Export to TorchScript
    model_scripted.save(tmp_save_model_file.name)  # Save

    file_loader.load_file(
        file_uri_src=LocalUri(tmp_save_model_file.name), file_uri_dst=save_to_path_uri
    )
    tmp_save_model_file.close()


def save_state_dict(model: torch.nn.Module, save_to_path_uri: Uri) -> None:
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    assert isinstance(
        model, torch.nn.Module
    ), "Can only save model of type torch.nn.Module"
    file_loader = FileLoader()

    tmp_save_model_file = tempfile.NamedTemporaryFile(delete=False)
    torch.save(model.state_dict(), tmp_save_model_file.name)

    file_loader.load_file(
        file_uri_src=LocalUri(tmp_save_model_file.name), file_uri_dst=save_to_path_uri
    )
    tmp_save_model_file.close()


def load_state_dict_from_uri(
    load_from_uri: Uri,
    device: torch.device = torch.device("cpu"),
) -> OrderedDict[str, torch.Tensor]:
    state_dict: OrderedDict[str, torch.Tensor]

    file_loader = FileLoader()
    tmp_file = file_loader.load_to_temp_file(load_from_uri)

    state_dict = torch.load(tmp_file.name, map_location=device)
    tmp_file.close()

    return state_dict


def load_scripted_model_from_uri(
    load_from_uri: Uri,
) -> torch.nn.Module:
    scripted_model: torch.jit.ScriptModule

    file_loader = FileLoader()
    tmp_file = file_loader.load_to_temp_file(load_from_uri)

    scripted_model = torch.jit.load(tmp_file.name)
    tmp_file.close()

    return scripted_model
