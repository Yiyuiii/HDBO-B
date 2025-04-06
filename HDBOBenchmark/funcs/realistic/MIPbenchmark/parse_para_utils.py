import os, sys
from pathlib import Path
import pandas as pd

ROOT_DIR = str(Path(os.path.realpath(__file__)).parent.parent)
sys.path.append(ROOT_DIR)
from .design_space import DesignSpace


def parse_int(row: pd.Series):
    name = row.name
    if len(row.range.split(",")) == 2:
        lb, ub = [int(x) for x in row.range.split(",")]
        if lb < ub:
            def_val = row.default
            return {"name": name, "type": "int", "lb": lb, "ub": ub, "default": def_val}
        else:
            values = [int(x) for x in row.range.split(",")]
            def_val = int(row.default)
            return {
                "name": name,
                "type": "cat",
                "categories": values,
                "default": def_val,
            }
    else:
        values = sorted([int(x) for x in row.range.split(",")])
        def_val = int(row.default)
        return {"name": name, "type": "cat", "categories": values, "default": def_val}


# def parse_bool(row: pd.Series):
#     name = row.name  # NOTE: name is index
#     def_val = row['default'].lower() == 'true'
#     return {'name': name, 'type': 'bool', 'default': def_val}


def parse_bool_cat(row: pd.Series):
    name = row.name  # NOTE: name is index
    categories = [0, 1]
    if row.default in ["True", "TRUE"]:
        def_val = 1
    elif row.default in ["False", "FALSE"]:
        def_val = 0
    else:
        RuntimeError(f"default value not recognised {row.default}")
    return {"name": name, "type": "bool", "categories": categories, "default": def_val}


def parse_real(row: pd.Series):
    name = row.name
    lb, ub = [float(x) for x in row.range.split(",")]
    def_val = row.default
    return {"name": name, "type": "num", "lb": lb, "ub": ub, "default": def_val}


def parse_char(row: pd.Series):
    name = row.name
    categories = list(row.range)
    def_val = row.default
    return {"name": name, "type": "cat", "categories": categories, "default": def_val}


def parse_str(row: pd.Series):
    name = row.name
    categories = list(row.range.split(","))
    def_val = row.default
    return {"name": name, "type": "cat", "categories": categories, "default": def_val}


parse_func = {}
parse_func["bool"] = parse_bool_cat
parse_func["char"] = parse_char
parse_func["str"] = parse_str
parse_func["int"] = parse_int
parse_func["real"] = parse_real


def parse_scip_para(file_path: str) -> DesignSpace:
    df = pd.read_csv(file_path, index_col=0)
    para_conf = []
    for i, row in df.iterrows():
        ptype = row.type.lower()
        if ptype not in parse_func.keys():
            print(f"Type {ptype} not supported")
        if ptype in parse_func.keys():
            para_conf.append(parse_func[ptype](row))

    def_val = {para["name"]: para["default"] for para in para_conf}
    design_space = DesignSpace().parse(para_conf)
    return design_space, def_val


if __name__ == "__main__":
    space, default = parse_scip_para("./conf/paras.csv")
    sample = space.sample(1)
    sample = pd.concat([pd.DataFrame([default]), sample], axis=0).reset_index(drop=True)
    print(sample.iloc[:, :5])
