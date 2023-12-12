from copy import deepcopy


class AdditiveDict(dict):

    def __init__(self, *args, **kwargs):
        on_error = kwargs.pop('on_error', 'ignore')

        super(AdditiveDict, self).__init__(*args, **kwargs)
        if isinstance(on_error, str):
            assert on_error in ['ignore', 'replace']
            self.on_error = getattr(self, f'{on_error}_func')
            # if on_error == 'ignore':
            #     self.on_error = lambda x, y: x
            # elif on_error == 'replace':
            #     self.on_error = lambda x, y: y
        elif hasattr(on_error, '__call__'):
            self.on_error = on_error
        else:
            raise ValueError('on_error must be a string or a callable object, not %r' % on_error)

    def __add__(self, other: dict):
        new_dict = deepcopy(self)
        new_dict.add(other, inplace=False)
        return new_dict

    def __iadd__(self, other):
        self.add(other, inplace=True)
        return self

    @staticmethod
    def ignore_func(old, new):
        return old

    @staticmethod
    def replace_func(old, new):
        return new

    def add(self, other, inplace=False):
        for k, v in other.items():
            if k in self.keys():
                try:
                    if inplace:
                        self[k] += v
                    else:
                        self[k] = self[k] + v
                except Exception as e:
                    self[k] = self.on_error(self[k], v)
            else:
                self[k] = v


if __name__ == '__main__':
    d = AdditiveDict('replace', {
        'num': 0,
        'list': [0],
        'str': 'hello ',
        'set': {1, 2, 3},
        'object': object(),
    })

    t = {
        'num': 2,
        'list': [1, 2, 3],
        'str': 'world',
        'set': {2, 3, 4, 5, },
        'object': object()
    }
    print(id(d), id(d['object']))
    v = d + t
    print(v, id(v), id(v['object']), id(t['object']))
    d += t
    print(d, id(d), id(d['object']), id(t['object']))
