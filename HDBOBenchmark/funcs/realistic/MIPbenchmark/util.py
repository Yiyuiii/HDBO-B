def un_gz(file_name: str, dest: str) -> str:
    import gzip

    """unzip .gz file

    Args:
        file_name (str): The file to unzip
        dest (str): unzip destination path/file name

    Returns:
        str: destination path/file name
    """

    f_name = file_name.replace(".gz", "")
    print(f"current file: {f_name}")
    g_file = gzip.GzipFile(file_name)
    if dest is not None:
        path = dest
    else:
        path = f_name
    open(path, "wb+").write(g_file.read())
    return path


def determine_file_type(file_name: str, file_type: str) -> bool:
    return file_name.split(".")[-1] == file_type


def is_gz(file_name) -> bool:
    return determine_file_type(file_name, "gz")


def is_mps(file_name) -> bool:
    return determine_file_type(file_name, "mps")


def is_npy(file_name) -> bool:
    return determine_file_type(file_name, "npy")


def get_keys_from_dict(dict, keys):
    from operator import itemgetter

    values = itemgetter(*keys)(dict)
    return values


def get_solutions(file_path="assets/miplib/miplib2017-v26.solu") -> dict:
    import math

    UNKNOWN = "=unkn="
    UNBOUND = "=unbd="
    INF = "=inf="

    with open(file_path, "r") as f:
        solutions = f.readlines()
        results = {}
        for i, solution in enumerate(solutions):
            solution_list = solution.split()
            result = {
                "type": solution_list[0],
                "filename": solution_list[1],
            }
            if result["type"] in [UNKNOWN, UNBOUND]:
                result["answer"] = None
            elif result["type"] == INF:
                result["answer"] = math.inf
            else:
                result["answer"] = solution_list[2]
            results[result["filename"]] = result

    return results


def download_miplib(path=r"assets/miplib/", target="easy-v11.test"):
    from pathlib import Path
    import requests

    base_dir = Path(path)
    base_dir.mkdir(parents=True, exist_ok=True)
    target_file = base_dir / target
    if not target_file.exists():
        print(f"Downloading {target} into {path}.")
        url = "https://miplib.zib.de/downloads/"
        r = requests.get(url + target)
        r.raise_for_status()
        with target_file.open("wb") as f:
            f.write(r.content)
    if ".zip" in target:
        import zipfile

        with zipfile.ZipFile(target_file.resolve(), mode="r") as archive:
            archive.extractall(path)
