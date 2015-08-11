def memoize(f):
    cache = {}

    def decorated(*args):
        if args not in cache:
            cache[args] = f(*args)
        else:
            print 'loading cached values for {}'.format(args)
        return cache[args]

    return decorated
