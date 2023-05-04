from matplotlib import pyplot as plt

cmap = plt.get_cmap("tab10")
display_colors = [cmap(i) for i in range(10)]


def get_multiplex_color(i: int):
    colors = ["blue", "green", "yellow", "red", "magenta", "cyan", "gray"]
    return colors[i]


def get_high_contrast_color(i: int):
    if i == 0:
        return "blue"
    elif i % 2 == 1:
        return "green"
    else:
        return "red"


__all__ = ["display_colors", "get_multiplex_color", "get_high_contrast_color"]
