def concatenate_lists_in_nested_dicts(*dicts):
    if len(dicts) == 0:
        return {}

    result = {}
    keys = set(dicts[0].keys())
    for dictionary in dicts:
        keys = keys.intersection(set(dictionary.keys()))

    for key in keys:
        values = [dictionary[key] for dictionary in dicts]
        if all(isinstance(value, list) for value in values):
            result[key] = sum(values, [])
        elif all(isinstance(value, dict) for value in values):
            result[key] = concatenate_lists_in_nested_dicts(*values)
        else:
            result[key] = values[0]

    return result


def average_lists_in_nested_dicts(dictionary):
    result = {}
    for key, value in dictionary.items():
        if isinstance(value, list):
            if not (len(value) > 0 and (
                    isinstance(value[0], int) or isinstance(value[0], float)  or isinstance(
                value[0], bool))):
                continue
            result[key] = sum(value) / len(value)
        elif isinstance(value, dict):
            result[key] = average_lists_in_nested_dicts(value)
        else:
            result[key] = value
    return result
