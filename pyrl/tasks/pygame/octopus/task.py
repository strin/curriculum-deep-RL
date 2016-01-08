from pyrl.tasks.task import Task
from pyrl.tasks.pygame.octopus.Lake.main import setup_screen
import pygame


class OctopusTask(Task):
    DISPLAY_NONE = 0
    DISPLAY_PYGAME = 1

    def __init__(self, display=DISPLAY_NONE):
        pygame.init()

        if display == OctopusTask.DISPLAY_PYGAME:
            setup_screen()


