from collections import OrderedDict
import json
import subprocess
import cStringIO
import urllib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cPickle as pickle
import util

__author__ = 'Kelvin Gu'


worksheet = 'nlp::millerjp'
site = 'http://localhost:18000'


def get_uuids():
    """
    Lists all bundle UUIDs in the worksheet
    """
    result = subprocess.check_output('cl work {}; cl ls -u'.format(worksheet), shell=True)
    uuids = result.split('\n')
    uuids = uuids[1:-1]  # trim non uuids
    return uuids


def get_file(uuid, path):
    """
    Gets the raw file content within a particular bundle at a particular path
    """
    # path should start with a leading slash
    url = '{}/api/bundles/filecontent/{}{}'.format(site, uuid, path)
    return cStringIO.StringIO(urllib.urlopen(url).read())


class Bundle(object):
    def __init__(self, uuid):
        self.uuid = uuid

    def __getattr__(self, item):
        """
        Load attributes: history, meta on demand
        """
        if item == 'history':
            try:
                value = pickle.load(get_file(self.uuid, '/history.cpkl'))
            except pickle.UnpicklingError:
                value = {}

        elif item == 'meta':
            try:
                value = json.load(get_file(self.uuid, '/meta.json'))
            except ValueError:
                value = {}

            # load codalab info
            fields = ('uuid', 'name', 'bundle_type', 'state', 'time', 'remote')
            cmd = 'cl work {}; cl info -f {} {}'.format(worksheet, ','.join(fields), self.uuid)
            result = subprocess.check_output(cmd, shell=True)
            result = result.split('\n')[1]
            info = dict(zip(fields, result.split()))
            value.update(info)

        elif item in ('stderr', 'stdout'):
            value = get_file(self.uuid, '/' + item).read()

        else:
            raise AttributeError(item)

        self.__setattr__(item, value)
        return value

    def __repr__(self):
        return self.uuid

    def load_img(self, img_path):
        """
        Return an image object that can be immediately plotted with matplotlib
        """
        img_file = get_file(self.uuid, img_path)
        return mpimg.imread(img_file)


def default_render(bundle):
    # print metadata
    print bundle
    print '{}/bundles/{}/'.format(site, bundle.uuid)
    for key in 'name state time remote uuid gb_used dataset augment ' \
               'warm_start wvec_dim batch_size step_size steps l2_reg ' \
               'mean_rank kbc positive_branch_factor'.split():
        try:
            val = bundle.meta[key]
            if key in ('uuid', 'remote'):
                val = val[:8]
        except KeyError:
            val = None

        print '{}: {}'.format(key, val)

    # convert history to nested dict
    history = util.NestedDict()
    for name, val in bundle.history.iteritems():
        history.set_nested(name, val)

    # plot history
    num_subplots = len(history)
    cols = 4  # 4 columns total
    rows = num_subplots / cols + 1

    fig = plt.figure()
    fig.subplots_adjust(wspace=0.0, hspace=0.0)  # no margin between subplots

    # Here we assume that history is only two levels deep
    for k, (subplot_name, trend_lines) in enumerate(history.iteritems()):
        plt.subplot(rows, cols, k + 1)
        plt.title(subplot_name)
        for name, (timestamps, values) in trend_lines.iteritems():
            plt.plot(timestamps, values, label=name)
        plt.legend()

    plt.show()


def old_render(bundle):
    # load images
    imgs = OrderedDict()

    # load images from directory
    def load_dir(directory):
        for img_name in ['objective', 'delta_norms', 'speed', 'map', 'mean_rank']:
            try:
                imgs[img_name] = bundle.load_img('{}/{}.png'.format(directory, img_name))
            except RuntimeError:
                pass

    load_dir('/kbc')

    fig = plt.figure()
    fig.subplots_adjust(wspace=0.0, hspace=0.0)  # no margin between subplots

    # 4 columns total
    num_images = len(imgs)
    cols = 4
    rows = num_images / cols + 1

    for k, (name, img) in enumerate(imgs.iteritems()):
        ax = plt.subplot(rows, cols, k + 1)
        ax.set_xlabel(name)
        plt.imshow(img)
        util.ticks_off()

    plt.show()


def report(uuids=None, reverse=True, limit=None, render=default_render):
    if uuids is None:
        uuids = get_uuids()

    if reverse:
        uuids = uuids[::-1]

    if limit is not None:
        uuids = uuids[:limit]

    for id in uuids:
        bundle = Bundle(id)
        if bundle.meta['bundle_type'] != 'run' or bundle.meta['state'] == 'queued':
            continue

        print '==' * 50
        render(bundle)
        old_render(bundle)  # added for backwards compatibility

    plt.close('all')
