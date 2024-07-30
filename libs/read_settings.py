import yaml  # to access settings

# Path to your YAML configuration file
config_file_path = "config.yaml"

# Retrieve info and return params
# To access specific values from the loaded configuration: use loaded_config["pricing"]["min"] etc. )
def read_config():

    # Read the configuration from the YAML file
    with open(config_file_path, "r") as config_file:
        loaded_config = yaml.load(config_file, Loader=yaml.FullLoader)

    # Access the configuration data
    if loaded_config is None:
        print("No configuration data found.")
        exit(0)

    return loaded_config
