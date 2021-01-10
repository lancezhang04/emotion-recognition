import matplotlib.pyplot as plt
import numpy as np
import json
import os


config = json.load(open("config/visualization_config.json"))
history = json.load(open(os.path.join(config["target_folder"], "train.log")))


for plot_config in config["plots"]:
    plt.figure(figsize=(10, 6))
    for field_name in plot_config["fields"]:
        field = history[field_name]

        field_range = plot_config["range"] if "range" in plot_config.keys() else None
        if field_range:
            field = field[field_range[0]: field_range[1]]

        smooth = plot_config["smooth"] if "smooth" in plot_config.keys() else None
        if smooth:
            field = np.array(field).flatten()
            field = field[:-(len(field) % smooth)]
            field = field.reshape(-1, smooth)
            field = np.mean(field, axis=1)
        else:
            field = np.mean(field, axis=1)
            if "calculate" in plot_config.keys():
                mode = plot_config["calculate"]
                if mode == "min":
                    minimum_idx = np.argmin(field)
                    minimum = field[minimum_idx]
                    plt.plot(
                        [0, len(field)], [minimum, minimum],
                        c="#ff6961", label=f"minimum: {minimum:.5f}\nidx: {minimum_idx:d}"
                    )
                elif mode == "max":
                    maximum_idx = np.argmax(field)
                    maximum = field[maximum_idx]
                    plt.plot(
                        [0, len(field)], [maximum, maximum],
                        c="#ff6961", label=f"maximum: {maximum:.5f}\nidx: {maximum_idx:d}"
                    )
                else:
                    raise ValueError

        with plt.style.context("seaborn-deep"):
            plt.plot(field, label=field_name)

    plt.legend(loc="best")
    plt.show()
