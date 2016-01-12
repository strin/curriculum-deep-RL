lines = file('octopus.out', 'r').readlines()
res = []

for line in lines:
    try:
        line = line.replace('\n', '')
        [x, y] = line.split(',')
        (x,y) = (int(x), int(y))
        res.append((x,y))
    except:
        pass

with open('2.coord.txt', 'w') as f:
    for (x, y) in res:
        f.write('octopus,%d,%d\n' % (x,y))




