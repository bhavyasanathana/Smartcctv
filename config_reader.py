def load_config(filename="config.txt"):
    config = {}
    with open(filename, "r") as file:
        for line in file:
            if "=" in line and not line.startswith("#"):
                key, value = line.strip().split("=")
                config[key] = value
    return config


if __name__ == "__main__":
    cfg = load_config()
    for k, v in cfg.items():
        print(k, ":", v)
