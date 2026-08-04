import yaml
from argparse import Namespace
from copy import deepcopy


def merge_dicts(base, override):
    result = deepcopy(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = deepcopy(value)

    return result


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]
    mode = config["mode"]

    if dataset_name not in config["datasets"]:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if mode not in config["modes"]:
        raise ValueError(f"Unknown mode: {mode}")

    dataset_config = deepcopy(config["datasets"][dataset_name])

    if mode not in dataset_config:
        raise ValueError(
            f"Mode '{mode}' is not configured for dataset '{dataset_name}'. "
            f"Currently, finetuning is only configured for micro_ct."
        )

    common_config = config.get("common", {})
    mode_config = config["modes"][mode]
    selected_mode_dataset_config = dataset_config.pop(mode)

    dataset_config.pop("training", None)
    dataset_config.pop("finetuning", None)
    dataset_config.pop("testing", None)

    top_level_config = {
        "dataset": dataset_name,
        "mode": mode,
        "dataroot": config["dataroot"],
        "checkpoints_dir": config["checkpoints_dir"],
    }

    if mode == "finetuning":
        top_level_config["pretrained_model_path"] = config["pretrained_model_path"]

    if mode == "testing":
        top_level_config["model_path"] = config["model_path"]
        top_level_config["output_dir"] = config["output_dir"]

    final_config = {}
    final_config = merge_dicts(final_config, common_config)
    final_config = merge_dicts(final_config, mode_config)
    final_config = merge_dicts(final_config, dataset_config)
    final_config = merge_dicts(final_config, selected_mode_dataset_config)
    final_config = merge_dicts(final_config, top_level_config)

    if final_config.get("max_dataset_size") == "inf":
        final_config["max_dataset_size"] = float("inf")

    return Namespace(**final_config)