def plot_func(Func):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # create function instance (dim=2)
    func = Func(dim=2)
    assert func.dim == 2

    # create meshgrids
    meshgrid_fineness = 100
    bounds = func.bound
    x = np.linspace(bounds[0][0], bounds[0][1], meshgrid_fineness)
    y = np.linspace(bounds[1][0], bounds[1][1], meshgrid_fineness)
    X, Y = np.meshgrid(x, y)

    # calculate the function value of each point on the grid
    Z = np.array(
        [func.evaluate(np.array([x, y]))["y"] for x, y in zip(np.ravel(X), np.ravel(Y))]
    )
    Z = Z.reshape(X.shape)

    # using Matplotlib to draw a surface graph, allow dragging for a 3D-view
    fig = plt.figure()
    ax = fig.gca(projection="3d")  # ax = fig.add_axes(Axes3D(fig))
    # ax.contour(X, Y, Z, cmap='jet', levels=20)
    ax.plot_surface(X, Y, Z, cmap=plt.cm.jet)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    plt.title(f"{Func.__name__} Function")
    plt.show()

    # save to local
    os.makedirs("./figs/", exist_ok=True)
    fig.savefig(f"./figs/{Func.__name__}.png")


if __name__ == "__main__":
    from HDBOBenchmark import Sphere as Func

    plot_func(Func)
