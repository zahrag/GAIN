
'''
    This scripts is to read logs saved while training GAIN
    '''


def get_logs(data_dir):
    logs = {}
    with open(data_dir + "/configs.txt") as fp:
        for i, line in enumerate(fp):
            a_string = line.strip()
            name = []
            info = []
            for word in a_string:
                info.append(word)
            for i, word in enumerate(info):
                if word == ':':
                    name = info[:i+1]
                    number = info[i+1:]

            name_ = ''.join(name)
            if number[0].isdigit():
                if name_ == 'dropout:' or name_ == 'weight_decay:' or name_ == 'Initial_Learning_Rate:':
                    numbers_ = float(''.join(number))
                else:
                    numbers_ = int(''.join(number))
            else:
                numbers_ = ''.join(number)

            logs[name_] = numbers_

    return logs


def get_sup_results(data_dir):
    val_results = {}
    with open(data_dir + "/val_stats.txt") as fp:
        for i, line in enumerate(fp):
            a_string = line.strip()
            info = []
            for j, word in enumerate(a_string):
                info.append(word)
                if word == ' ' or j == len(a_string)-1:
                    for i, letter in enumerate(info):
                        if letter == '=':
                            name = info[:i + 1]
                            number = info[i + 1:]
                            info = []
                            name_ = ''.join(name)
                            if number[0].isdigit():
                                numbers_ = float(''.join(number))
                                val_results[name_] = numbers_

    test_results = {}
    with open(data_dir + "/test_stats.txt") as fp:
        for i, line in enumerate(fp):
            a_string = line.strip()
            info = []
            for j, word in enumerate(a_string):
                info.append(word)
                if word == ' ' or j == len(a_string) - 1:
                    for i, letter in enumerate(info):
                        if letter == '=':
                            name = info[:i + 1]
                            number = info[i + 1:]
                            info = []
                            name_ = ''.join(name)
                            if number[0].isdigit():
                                numbers_ = float(''.join(number))
                                test_results[name_] = numbers_

    return val_results, test_results






