import os
import argparse
import configs

print "WARNING: USING EXISTING CODE. Upload a new bundle if there are changes"

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

options = ""
for var, val in config.iteritems():
    options += '--' + var + ' ' + str(val) + ' '


command = 'clb ex %s \'%s\' %s %s %d %dg ' % (args.task, options, args.name, args.queue,
                                              args.cpus, args.memory)
print command
os.system(command)
