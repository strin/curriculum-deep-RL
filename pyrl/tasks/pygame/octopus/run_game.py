#! /usr/bin/env python
import os
from Lake.main import start_game, SyncEvent, AsyncEvent
from threading import Thread

level = os.environ.get('level')
level = int(level) if level and level.isdigit() else level
event = os.environ.get('event')
event = event if event else 'sync'

print 'event type', event
print 'start level', level

if event == 'sync':
    EventType = SyncEvent
elif event == 'async':
    EventType = AsyncEvent
else:
    raise Exception('unrecognized event type')

start_game(event=EventType(), level=level)
