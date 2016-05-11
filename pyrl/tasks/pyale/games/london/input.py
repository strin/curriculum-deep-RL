# London's burning
# a game by Adam Binks

import pygame, sys, outro
from pygame.locals import *

class Input:
    """A class to handle input accessible by all other classes"""
    constrainMouseMargin = 30
    def __init__(self):
        self.pressedKeys = []
        self.mousePressed = False
        self.mouseUnpressed = False
        self.mousePos = [0, 0]
        self.num_frames = 0
        

    def get(self, data,  constrainMouse=False, inOutro=False):
        """Update variables - mouse position and click state, and pressed keys"""
        self.mouseUnpressed = False
        self.unpressedKeys = []
        self.justPressedKeys = []
        self.num_frames += 1


        for event in pygame.event.get():
            if self.num_frames % 10: # controls CD of bombing.
                self.mousePressed = False
                self.mouseUnpressed = 1
                continue
            # replace mouse events with key events.
            if event.type == KEYDOWN:
                angle = ord(event.key) - ord('0') # 0 - 180
                if angle >= 30:
                    angle = 29
                width = pygame.display.get_surface().get_width()
                height = pygame.display.get_surface().get_height()
                width = int(float(width) * angle / 30)
                height = height - 300

                self.mousePos = [width, height]
                self.mousePressed = 1
                self.mouseUnpressed = False
            elif event.type == KEYUP:
                self.mousePressed = False
                self.mouseUnpressed = 1

            '''
            if event.type == MOUSEMOTION:
                self.mousePos = list(event.pos)
            elif event.type == MOUSEBUTTONDOWN:
                self.mousePressed = event.button
                self.mouseUnpressed = False
            elif event.type == MOUSEBUTTONUP:
                self.mousePressed = False
                self.mouseUnpressed = event.button
            '''

            if event.type == KEYDOWN:
                if event.key not in self.pressedKeys:
                    self.justPressedKeys.append(event.key)
                self.pressedKeys.append(event.key)
            elif event.type == KEYUP:
                for key in self.pressedKeys:
                    if event.key == key:
                        self.pressedKeys.remove(key)
                    self.unpressedKeys.append(key)
            elif event.type == QUIT:
                pygame.event.post(event)

        if constrainMouse:
            self.constrainMouse()
        if not inOutro:
            self.checkForQuit(data)


    def constrainMouse(self):
        """Set the cursor position to a little inside the edge of the window if it goes outside it"""
        for axis in [0, 1]:
            if self.mousePos[axis] < Input.constrainMouseMargin:
                self.mousePos[axis] = Input.constrainMouseMargin
            if self.mousePos[axis] > Input.winSize[axis] - Input.constrainMouseMargin:
                self.mousePos[axis] = Input.winSize[axis] - Input.constrainMouseMargin
        pygame.mouse.set_pos(self.mousePos)



    def checkForQuit(self, data):
        """Terminate if QUIT events"""
        for event in pygame.event.get(QUIT): # get all the QUIT events
            self.showOutro(data) # show the outro if any QUIT events are present
        if K_ESCAPE in self.unpressedKeys:
            self.showOutro(data)


    def showOutro(self, data):
        """Show an outro screen"""
        outro.showOutro(data)


    def terminate(self):
        """Safely end the program"""
        pygame.quit()
        sys.exit()
