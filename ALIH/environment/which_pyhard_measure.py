from pathlib import Path


def which_pyhard_measure(measure='LSC'):
    import yaml
    with open(Path('.') / 'strategies' / 'pyHard' / 'config-template.yaml') as file:
        configs_list = yaml.load(file, Loader=yaml.FullLoader)

        if measure == 'LSC':
            configs_list['measures_list'] = ['LSC']
        elif measure == 'Harmfulness':
            configs_list['measures_list'] = ['Harmfulness']
        elif measure == 'Usefulness':
            configs_list['measures_list'] = ['Usefulness']
        elif measure == 'U_H':
            configs_list['measures_list'] = ['Harmfulness', 'Usefulness']
        elif measure == 'N2':
            configs_list['measures_list'] = ['N2']
        elif measure == 'F3':
            configs_list['measures_list'] = ['F3']

    with open(Path('.') / 'strategies' / 'pyHard' / 'pyhard_files' / 'config.yaml', 'w') as file:
        yaml.dump(configs_list, file)