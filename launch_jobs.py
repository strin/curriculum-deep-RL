import os
import argparse
import copy
import configs

print "WARNING: USING EXISTING CODE. UPLOAD A NEW BUNDLE IF THERE ARE CHANGES"

parser = argparse.ArgumentParser()

# task arguments
parser.add_argument('task')
parser.add_argument('config')
parser.add_argument('name')
parser.add_argument('-q', '--queue', default='nlp')
parser.add_argument('-c', '--cpus', type=int, default=8)
parser.add_argument('-m', '--memory', type=int, default=5)
args = parser.parse_args()

config = getattr(configs, args.config)

options = [""]
for var, val in config.iteritems():
    if isinstance(val, list):
        new_options = []
        for v in val:
            next = copy.deepcopy(options)
            for idx in xrange(len(options)):
                next[idx] += '--' + var + ' ' + str(v) + ' '

            new_options += next
        options = new_options
    else:
        for idx in xrange(len(options)):
            options[idx] += '--' + var + ' ' + str(val) + ' '


for opt in options:
    command = 'clb ex %s \'%s\' %s %s %d %dg ' % (args.task, opt, args.name, args.queue,
                                                  args.cpus, args.memory)
    print command
    os.system(command)
