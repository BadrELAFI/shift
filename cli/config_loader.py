import yaml


class ConfigLoader:
    @staticmethod
    def load_config():
        try:
            with open("config.yaml", "r") as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError as e:
            raise RuntimeError("Configuration file 'config.yaml' not found.") from e
