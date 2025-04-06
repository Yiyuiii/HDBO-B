import numpy as np
import matplotlib.pyplot as plt
import os


def get_linestyle_list(num: int) -> list:
    if num == 1:
        return [("black", "-")]
    if num <= 6:
        return [
            ("b", "-"),
            ("r", "-"),
            ("g", "-"),
            ("m", "--"),
            ("c", "--"),
            ("purple", "--"),
            ("black", "-"),
        ]
    else:
        return [
            ("b", "-"),
            ("r", "-"),
            ("g", "-"),
            ("m", "-"),
            ("c", "-"),
            ("purple", "-"),
            ("b", "--"),
            ("r", "--"),
            ("g", "--"),
            ("m", "--"),
            ("c", "--"),
            ("purple", "--"),
            ("black", "-"),
            ("black", "--"),
        ]


def linear_interpolation(x, x1, y1, x2, y2):
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1)


def cal_std_data(*data):
    """
    data: {'x':array, 'y':array}
    """
    n_data = len(data)
    assert n_data > 0  # no results?

    x = list()
    mean = list()
    std = list()
    std_up = list()
    std_down = list()
    len_data = [len(data[i]["x"]) for i in range(n_data)]
    p = [0 for _ in range(n_data)]
    flag_end = [False for _ in range(n_data)]
    while True:
        # get next x
        next_x = list()
        for i in range(n_data):
            if not flag_end[i]:
                next_x.append(data[i]["x"][p[i]])
            else:
                continue
        if len(next_x) == 0:
            break

        next_x = np.min(next_x)
        x.append(next_x)
        # get next y
        next_y = list()
        for i in range(n_data):
            if (data[i]["x"][p[i]] > next_x) and (p[i] > 0):
                data_y = linear_interpolation(
                    next_x,
                    data[i]["x"][p[i] - 1],
                    data[i]["y"][p[i] - 1],
                    data[i]["x"][p[i]],
                    data[i]["y"][p[i]],
                )
            elif (data[i]["x"][p[i]] < next_x) and (p[i] < len_data[i] - 1):
                data_y = linear_interpolation(
                    next_x,
                    data[i]["x"][p[i]],
                    data[i]["y"][p[i]],
                    data[i]["x"][p[i] + 1],
                    data[i]["y"][p[i] + 1],
                )
            else:
                data_y = data[i]["y"][p[i]]
            next_y.append(data_y)
        next_y = np.array(next_y)
        # cal mean, std
        mean.append(next_y.mean())
        std.append(next_y.std())
        std_up.append(
            np.sqrt(np.square(next_y[np.where(next_y >= mean[-1])] - mean[-1]).mean())
        )
        std_down.append(
            np.sqrt(np.square(next_y[np.where(next_y <= mean[-1])] - mean[-1]).mean())
        )
        # next p[i]
        for i in range(n_data):
            if data[i]["x"][p[i]] == next_x:
                if p[i] == len_data[i] - 1:
                    flag_end[i] = True
                else:
                    p[i] += 1
    return {
        "x": np.array(x),
        "y": np.array(mean),
        "std": np.array(std),
        "std_up": np.array(std_up),
        "std_down": np.array(std_down),
        "name": data[0]["name"],
    }


def plt_init(style="bmh", **kwargs):
    # print(plt.style.available)
    plt.style.use(style)
    # plt.rc('font', family='STFangsong')


def sub_fig(fig, xlabel="x", ylabel="y", **kwargs):
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def plot(data, ax, color="b", linestyle="-", **kwargs):
    ax.plot(data["x"], data["y"], linestyle, label=data["name"], color=color, alpha=0.6)
    if "std_up" in data.keys():
        plt.gca().fill_between(
            data["x"],
            data["y"] - data["std_down"],
            data["y"] + data["std_up"],
            facecolor=color,
            alpha=0.2,
        )
    elif "std" in data.keys():
        plt.gca().fill_between(
            data["x"],
            data["y"] - data["std"],
            data["y"] + data["std"],
            facecolor=color,
            alpha=0.2,
        )


def test():
    cwd = "./figs"

    x = np.array(range(100))
    y1 = np.random.randn(100)
    y2 = np.random.randn(100)

    a = cal_std_data(
        {"x": x, "y": y1, "name": "data1"}, {"x": x, "y": y2, "name": "data2"}
    )

    plt_init(style="bmh")
    Color_list = ("b", "r", "g", "m")
    fig = plt.figure()
    ax = sub_fig(fig, xlabel="x", ylabel="y")
    plot(a, ax, color=Color_list[0])
    ax.legend(loc="best", frameon=True, facecolor="white")
    plt.tight_layout()
    try:
        plt.show()
    except:
        os.makedirs(cwd, exist_ok=True)
        save_img_path = os.path.join(cwd, "test.png")
        if os.path.exists(save_img_path):
            os.remove(save_img_path)
        fig.savefig(save_img_path, bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    test()
