from .logging import logger
import numpy as np
import torch
import os
import random


def get_attr(o, n, v):
    attr = getattr(o, n, None)
    if attr is None:
        logger.debug(f"{n} not found, default: {v}")
        attr = v
    return attr


def set_seed(seed=None, torch_deterministic=False):
    if (seed is None) and torch_deterministic:
        seed = 42
    elif seed is None:
        seed = np.random.randint(0, 10000)
    logger.info(f"Setting seed: {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_deterministic(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed


def get_torch_whl(cuda_version="11.8", python_version="3.8", os_version="linux"):
    import requests
    from os.path import join
    from bs4 import BeautifulSoup

    # 构建请求的URL
    request_url = "https://download.pytorch.org/whl/torch/"
    base_url = "https://download.pytorch.org"

    # 发送请求
    response = requests.get(request_url)
    if response.status_code != 200:
        return "Unable to access PyTorch download page, please check your network connection."

    # 解析页面内容
    soup = BeautifulSoup(response.content, "html.parser")
    links = soup.find_all("a")

    # 构建筛选条件
    os_filter = "linux" if os_version.lower() == "linux" else "win"
    python_filter = f"cp{python_version.replace('.', '')}"
    cuda_filter = (
        f"cu{cuda_version.replace('.', '')}" if cuda_version != "cpu" else "cpu"
    )

    # 筛选符合条件的链接
    filtered_links = []
    for link in links:
        href = link.get("href")
        if os_filter in href and python_filter in href and cuda_filter in href:
            filtered_links.append(href)

    # 如果没有找到符合条件的链接
    if not filtered_links:
        return "No torch whl file found that meets the specified criteria."

    # 返回最新的链接
    latest_link = sorted(filtered_links, reverse=True)[0]
    return base_url + latest_link


if __name__ == "__main__":
    print(get_torch_whl(cuda_version="11.4", python_version="3.9", os_version="linux"))
