from pathlib import Path

def pyhard_config(section, filename='strategies.config'):
    from configparser import ConfigParser

    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(Path('.') / 'strategies' / 'pyHard' / filename)
    # get section, default to postgresql
    strategy = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            strategy[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))

    # transformando texto em bool
    strategy['ascending'] = list(map(lambda x: bool(0 if x == "False" else 1), strategy['ascending'].split(',')))
    strategy['sortby'] = strategy['sortby'].split(',')

    print(strategy)

    return strategy