from collections.abc import MutableMapping

# https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
def flatten(d, parent_key="", sep="$"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
def unflatten(d, sep="$"):
    ud = {}
    for k, v in d.items():
        context = ud
        for sub_key in k.split(sep)[:-1]:
            if sub_key not in context:
                context[sub_key] = {}
            context = context[sub_key]
        context[k.split(sep)[-1]] = v
    return ud
