import argparse
import yaml
import json
import subprocess
from jinja2 import Environment, BaseLoader

env = Environment(loader=BaseLoader())
env.filters['json'] = json.dumps

def get_value_from_yaml(file_path, variable_path, sep):
    """
    Reads a YAML file and returns the value at the given path.

    Args:
        file_path (str): Path to the YAML file
        variable_path (str): Path to the variable (for example: foo.bar)
        sep (str): Address divider character (default: '.')

    Returns:
        tuple: The value and the entire YAML data as dictionaries
    """
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
        if variable_path == sep:
            value = data
        else:
            value = _get_value_by_path(data, variable_path.split(sep))
        return value, data

def _get_value_by_path(data, keys):
    for key in keys:
        data = data[key]
    return data


def main():
    """
    Command line interface for accessing a specific variable in a YAML file.

    Can be used to print the value of a variable or to run a script
    that is stored in the YAML file.

    Args:
        conf_path (str): Path to the YAML file
        script (str): Script to run
        -p PATH (str): Path to the variable to be accessed (for example: foo.bar)
        -s SEPARATOR (str): Address divider character (default: '.')

    Examples:
        python monkey.py conf.yaml echo -p foo.bar
        python monkey.py conf.yaml keys -p foo
        python monkey.py conf.yaml my_script
    """
    parser = argparse.ArgumentParser(description="YAML dosyasındaki belirli bir değişkene erişme")
    parser.add_argument("conf_path", help="YAML file path")
    parser.add_argument("script", help="Script to run")
    parser.add_argument("-p", "--path", help="Path to the variable to be accessed (for example: foo.bar)")
    parser.add_argument("-s", "--separator", help="Address divider character", default='.')

    args = parser.parse_args()

    data, full_data = None, None
    if args.path is not None:
        data, full_data = get_value_from_yaml(args.conf_path, args.path, args.separator)
    
    if args.script == 'keys':
        print([i for i in data])
        exit()
    elif args.script == 'echo':
        print(data)
        exit()
    scripts, full_data = get_value_from_yaml(args.conf_path, f"scripts{args.separator}{args.script}", args.separator)
    context = {'data': data, 'root': full_data}
    for script in scripts:
        template = env.from_string(str(script))
        command = template.render(context)
        print("$",command)
        subprocess.run(command, shell=True, encoding='utf-8')


if __name__ == "__main__":
    main()