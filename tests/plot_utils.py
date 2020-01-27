colors = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]


class number_formatter(float):
    def __repr__(self):
        str = "%.1f" % (self.__float__(),)
        if str[-1] == "0":
            return "%.0f" % self.__float__()
        else:
            return "%.1f" % self.__float__()
