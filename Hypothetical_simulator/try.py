import numpy as np
a = {'I' : [{'love': np.int_([1,2,4])}], 'Love' : {'I' : {'love': np.int64(5)}}}


def myconverter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def convert_dict(dicts):
    if isinstance(dicts, dict) == False and isinstance(dicts, list)== False and \
            isinstance(dicts, np.ndarray) == False:
        print('Not dicts', dicts)
        return myconverter(dicts)

    else:
        print('dicts', dicts)
        if isinstance(dicts, list) or isinstance(dicts, np.ndarray):
            enu_stuff = enumerate(dicts)
        else:
            enu_stuff = dicts.items()
        for key, value in enu_stuff:
            print('key', key)
            dicts[key] = myconverter(dicts[key])
            print(type(dicts[key]))
            if isinstance(dicts[key], dict) or isinstance(dicts[key], list) or isinstance(dicts[key], np.ndarray):
                dicts[key] = convert_dict(dicts[key])

        return dicts



a = convert_dict(a)
print(a)
print(type(a['I'][0]['love']), type(a['Love']['I']['love']))
