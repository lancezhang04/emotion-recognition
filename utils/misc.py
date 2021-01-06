import os


def calculate_project_size(extensions=("json", "py")):
    total_lines = 0

    for root, dirs, files in os.walk("."):
        for file in files:
            if file.split(".")[-1] in extensions:
                with open(os.path.join(root, file), "r") as f:
                    file_length = len(f.readlines())
                    print(file.ljust(30), file_length)
                    total_lines += file_length

    return total_lines


def format_description(history, iters=1):
    return ", ".join(["%s: %.5f" % (k, v / iters) for k, v in history.items()])
