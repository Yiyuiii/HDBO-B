from tensorboard.backend.event_processing import event_accumulator
import json
import os
import re
import pprint


def load_tensorboard_file(tensorboard_path):
    ea = event_accumulator.EventAccumulator(tensorboard_path)
    ea.Reload()
    return ea


def read_tensorboard_data(ea, val_name):
    """读取tensorboard数据，
    tensorboard_path是tensorboard数据地址val_name是需要读取的变量名称"""
    data_dict = {
        "step": list(),
        "value": list(),
        "wall_time": list(),
    }
    data_list = ea.scalars.Items(val_name)
    for scalar in data_list:
        data_dict["step"].append(scalar.step)
        data_dict["value"].append(scalar.value)
        data_dict["wall_time"].append(scalar.wall_time)
    return data_dict


def path_find_file(path, pattern):
    files_name = os.listdir(path)
    for name in files_name:
        file = os.path.join(path, name)
        if os.path.isfile(file) and (re.match(pattern, name) is not None):
            return name
    return None


def log_scalars(logger, result: dict, global_step):
    if logger is not None:
        for n, v in result.items():
            logger.add_scalar(n, v, global_step=global_step)


def log_hparam(args, dir, filename="hparam.json"):
    hparam_dict = _get_hparam_dict(args)
    pprint.pprint(hparam_dict)
    hparam_json = json.dumps(
        hparam_dict, sort_keys=True, indent=4, separators=(",", ": ")
    )
    with open(os.path.join(dir, filename), "w") as f:
        f.write(hparam_json)
    del hparam_dict, hparam_json


def _get_hparam_dict(args):
    def get_hparam_dict(hparam_dict, item, name):
        if isinstance(item, (bool, str, float, int)):
            hparam_dict[name] = item
        elif isinstance(item, (dict,)):
            for key in item:
                get_hparam_dict(hparam_dict, item[key], f"{name}.{key}")
        elif hasattr(item, "__name__"):
            hparam_dict[name] = item.__name__
        elif item is None:
            hparam_dict[name] = "None"
        else:
            hparam_dict[name] = item.__class__.__name__

    hparam_dict = {}
    for name in args.__dict__:
        get_hparam_dict(hparam_dict, getattr(args, name), name)
    return hparam_dict
