'''Game main module.

Contains the entry point used by the run_game.py script.

Feel free to put all your game code here, or in other modules in this "gamelib"
package.
'''

# just the imports for the game
import data
import pygame
import sys
import math
import os
import time
import random
from pygame.locals import *

# Here are the constants for the game
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 100, 255)
(SCREEN_WIDTH, SCREEN_HEIGHT) = (640, 480)
clock = pygame.time.Clock()

# this function loads an image from the data folder into pygame
def loadIm(name):
    file = data.filepath(name)
    try:
        image = pygame.image.load(file)
        return image.convert_alpha()
    except:
        name = 'error.png'
        print 'Could not load:', file
        file = data.filepath(name)
        image = pygame.image.load(file)
        return image.convert_alpha()


class SoundEffect:
    def play(self, name, fadein, loop=0):
        file = data.filepath(name)
        self.name = name
        try:
            self.effect = pygame.mixer.Sound(file)
            self.effect.play(loops=loop, fade_ms=fadein)
        except:
            print "Could not play:", file
    def stop(self, fadeout):
        self.effect.fadeout(fadeout)




#Here is all the object classes for the game


class Finish:
    def __init__(self, end):
        self.end = end
        self.frame = 1
        self.finished = False

    def update(self):
        if self.frame == self.end:
            self.finished = True
        self.frame += 1



class Octopus(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.location = 0
        self.images = [loadIm("OctopusAnimation/1.png"), loadIm("OctopusAnimation/2.png"), loadIm("OctopusAnimation/3.png"), loadIm("OctopusAnimation/4.png"), loadIm("OctopusAnimation/5.png")]
        self.image = self.images[self.location]
        self.type = "octopus"
        #set whether the player can move these directions
        self.up = True
        self.down = True
        self.left = True
        self.right = True
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.fallspeed = 0
        self.rect.topleft = (self.x, self.y)

    def update(self, direction):
        self.location += direction/4.0
        self.location = self.location%4
        self.image = self.images[int(self.location)]
        self.y += self.fallspeed
        self.fallspeed += 0.1
        if self.x > 640 - self.rect.w:
            self.x = 640 - self.rect.w
        if self.x < 0:
            self.x = 0
        self.rect.topleft = (self.x, self.y)



class Collider(pygame.sprite.Sprite):
    def __init__(self, (x, y), (ox, oy), (w, h), direction):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image = pygame.Surface((w, h), SRCALPHA, 32)
        self.image.convert_alpha()
        #debugging
        #self.image.fill(BLACK)
        self.rect = Rect(x+ox, y+oy, w, h)
        self.ox, self.oy = ox, oy
        self.type = direction
        self.direction = direction
        self.x = x
        self.y = y
        self.rect.topleft = (self.x, self.y)

    def update(self):
        self.rect.topleft = (self.x+self.ox, self.y+self.oy)




class Animation(pygame.sprite.Sprite):
    def __init__(self, x1, y1, x2, y2, frames, image):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image = loadIm(image.strip())
        self.rect = self.image.get_rect()
        self.type = "animation"
        self.x = x1
        self.y = y1
        # the current frame
        self.frame = 1
        #the total frames
        self.frames = frames
        self.xSpeed = (x2 - x1) / float(frames)
        self.ySpeed = (y2 - y1) / float(frames)
        self.rect.topleft = (int(self.x), int(self.y))

    def update(self):
        if self.frame <= self.frames:
            self.x += self.xSpeed
            self.y += self.ySpeed
            self.rect.topleft = (int(self.x), int(self.y))
        self.frame += 1


class Ship(pygame.sprite.Sprite):
    def __init__(self, x, y, speed):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.images = [loadIm("shipl.png"), loadIm("shipr.png")]
        self.image = self.images[0]
        self.rect = self.image.get_rect()
        self.type = "ship"
        self.x = x
        self.y = y
        self.location = 1
        self.direction = -1
        self.location += self.direction
        self.speed = speed
        self.count = 0
        self.shoot = False
        self.rect.topleft = (self.x, self.y)

    def update(self):
        self.image = self.images[self.location]
        self.shoot = False
        if self.count == self.speed:
            self.shoot = True
            self.count = 0
        self.count += 1
        self.x += self.direction
        self.rect.topleft = (int(self.x), int(self.y))



class Boss(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.images = [loadIm("Spaceshipl.png"), loadIm("Spaceshipr.png")]
        self.image = self.images[0]
        self.rect = self.image.get_rect()
        self.type = "boss"
        self.x = x
        self.y = y
        self.location = 1
        self.direction = -1
        self.location += self.direction
        self.health = 3
        # for bombs
        self.speed = 51
        self.count = 0
        self.shoot = False
        # for packages
        self.speed2 = 400
        self.count2 = 0
        self.shoot2 = False

        self.rect.topleft = (self.x, self.y)

    def update(self):
        self.image = self.images[self.location]
        self.shoot = False
        if self.count == self.speed:
            self.shoot = True
            self.count = 0
        self.shoot2 = False
        if self.count2 == self.speed2:
            self.shoot2 = True
            self.count2 = 0
        self.count += 1
        self.count2 += 1
        self.x += self.direction
        self.rect.topleft = (int(self.x), int(self.y))




class Insert(pygame.sprite.Sprite):
    def __init__(self, x, y, start, end, image):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image = pygame.Surface((1,1), SRCALPHA, 32)
        self.image.convert_alpha()
        self.image2 = loadIm(image.strip())
        self.rect = self.image2.get_rect()
        self.type = "insert"
        self.x = x
        self.y = y
        # the current frame
        self.frame = 1
        #the start and end frames
        self.start = start
        self.end = end
        self.rect.topleft = x,y

    def update(self):
        if self.frame == self.start:
            self.image = self.image2
        if self.frame == self.end:
            self.image = pygame.Surface((1,1), SRCALPHA, 32)
            self.image.convert_alpha()
        self.frame += 1



class Dirt(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image = loadIm("DirtBlock.png")
        self.rect = self.image.get_rect()
        self.type = "dirt"
        self.x = x
        self.y = y
        self.rect.topleft = (self.x, self.y)



class Stone(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.images = [loadIm("Stone1.png"), loadIm("Stone2.png"), loadIm("Stone3.png"), loadIm("Stone4.png")]
        self.image = self.images[random.randint(0,3)]
        self.rect = self.image.get_rect()
        self.type = "stone"
        self.x = x
        self.y = y
        self.rect.topleft = (self.x, self.y)


class PlaySound:
    def __init__(self, start, finish, fadein, fadeout, name):
        self.name = name.strip()
        self.fadein = fadein
        self.fadeout = fadeout
        self.sound = SoundEffect()
        self.start = start
        self.finish = finish
        self.frame = 1
        self.playing = False
        if finish < start:
            raise("finish is less than start")

    def update(self):
        if self.frame == self.start:
            self.playing = True
            self.sound.play(self.name, self.fadein)
        if self.frame == self.finish:
            self.playing = False
            self.sound.stop(self.fadeout)
        self.frame += 1


class PlayMusic:
    def __init__(self, name):
        if name != None:
            self.name = name.strip()
            self.sound = SoundEffect()
            self.sound.play(self.name, 0, loop  =-1)
            self.playing = True
        else:
            self.playing = False

    def newSong(self, name):
        if name != None and self.playing == True:
            self.sound.stop(0)
            self.name = name.strip()
            self.sound = SoundEffect()
            self.sound.play(self.name, 0, loop  =-1)
            self.playing = True
        elif name != None:
            self.name = name.strip()
            self.sound = SoundEffect()
            self.sound.play(self.name, 0, loop  =-1)
            self.playing = True




class TP(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.images = []
        # which image is showing
        self.location = 0
        #load all 40 images
        for i in xrange(1, 41):
            self.images.append(loadIm("TPAnimation/" + str(i) + ".png"))
        self.image = self.images[0]
        self.rect = self.image.get_rect()
        self.type = "teleporter"
        self.x = x
        self.y = y
        self.rect.topleft = (self.x, self.y)

    def update(self):
        self.location += 1
        self.location = self.location % 40
        self.image = self.images[int(self.location)]
        self.rect.topleft = (self.x, self.y)


class Sphere(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image = loadIm("sphere.png")
        self.rect = self.image.get_rect()
        self.type = "sphere"
        self.x = x+1
        self.y = y+2
        self.rect.topleft = (self.x, self.y)
    def update(self):
        pass



class Explosion(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.images =[loadIm("Explosion/1.png"),loadIm("Explosion/2.png"),loadIm("Explosion/3.png"),loadIm("Explosion/4.png")]
        self.image = self.images[0]
        self.location = 0
        self.rect = self.image.get_rect()
        self.type = "explosion"
        self.x = x+1
        self.y = y+2
        self.rect.center = (self.x, self.y)
    def update(self):
        if self.location <= 3:
            self.image = self.images[int(self.location)]
        else:
            self.remove()
        self.location += 0.5
        self.rect = self.image.get_rect()
        self.rect.center = (self.x, self.y)




class Launcher(pygame.sprite.Sprite):
    def __init__(self, x, y, speed):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image = loadIm("Launcher.png")
        self.rect = self.image.get_rect()
        self.type = "launcher"
        self.x = x
        self.y = y
        self.speed = speed
        self.count = 0
        self.shoot = False
        self.rect.topleft = (self.x, self.y)

    def update(self):
        self.shoot = False
        if self.count == self.speed:
            self.shoot = True
            self.count = 0
        self.count += 1



class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.images = [loadIm("Bullet/1.png"), loadIm("Bullet/2.png"), loadIm("Bullet/3.png")]
        self.image = self.images[0]
        self.rect = self.image.get_rect()
        self.type = "bullet"
        self.x = x
        self.y = y
        self.speed = 7
        self.count = 0
        self.shoot = False
        self.rect.topleft = (self.x, self.y)

    def update(self):
        self.x -= self.speed
        self.rect.topleft = (self.x, self.y)



class Rocket(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image = loadIm("Rocket.png")
        self.rect = self.image.get_rect()
        self.type = "rocket"
        self.x = x
        self.y = y
        self.speed = 12
        self.count = 0
        self.shoot = False
        self.rect.topleft = (self.x, self.y)

    def update(self):
        self.y -= self.speed
        self.rect.topleft = (self.x, self.y)


class Bomb(pygame.sprite.Sprite):
    def __init__(self, x, y, xSpeed):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image = loadIm("Bomb.png")
        self.rect = self.image.get_rect()
        self.type = "bomb"
        self.x = x
        self.y = y
        self.xSpeed = xSpeed
        self.speed = 0
        self.count = 0
        self.shoot = False
        self.rect.center = (self.x, self.y)

    def update(self):
        self.y += self.speed
        self.x += self.xSpeed
        self.rect.midtop = (self.x, self.y)
        self.speed += 0.1
        self.xSpeed /= 1.02



class Package(pygame.sprite.Sprite):
    def __init__(self, x, y, xSpeed):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image = loadIm("Package.png")
        self.rect = self.image.get_rect()
        self.type = "package"
        self.x = x
        self.y = y
        self.falling = True
        self.xSpeed = xSpeed
        self.speed = 0
        self.count = 0
        self.shoot = False
        self.rect.center = (self.x, self.y)

    def update(self):
        self.y += self.speed
        self.x += self.xSpeed
        self.rect.midtop = (self.x, self.y)
        if self.falling == True:
            self.speed += 0.03
        else:
            self.speed = 0
            self.xSpeed = 0
        self.xSpeed /= 1.02

class Grass(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image = loadIm("GrassBlock.png")
        self.rect = self.image.get_rect()
        self.type = "grass"
        self.x = x
        self.y = y
        self.rect.topleft = (self.x, self.y)




class Sky(pygame.sprite.Sprite):
    def __init__(self, image):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image = loadIm(image)
        self.rect = self.image.get_rect()
        self.type = "sky"
        self.x = 0
        self.y = 0
        self.rect.topleft = (self.x, self.y)

    def update(self, image):
        self.image = loadIm(image)



class Water(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image = loadIm("WaterSmall.png")
        self.type = "water"
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rect.topleft = (self.x, self.y)



class Waves(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image = loadIm("WavesSmall.png")
        self.type = "waves"
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rect.topleft = (self.x, self.y)




# controls the menu
def Menu(screen):
    background = loadIm("Menu.png")
    start = loadIm("Start.png")
    quit = loadIm("Quit.png")
    # draw the menu
    screen.blit(background, (0,0))
    screen.blit(start, (100,220))
    screen.blit(quit, (100,290))
    # this monitors which menu item is selected
    selected = 0
    pygame.display.update()
    #when this is false the menu closes
    displayMenu = True
    while displayMenu:
        # loads the appropriate images
        if selected == 0:
            start = loadIm("StartSelected.png")
            quit = loadIm("Quit.png")
        elif selected == 1:
            start = loadIm("Start.png")
            quit = loadIm("QuitSelected.png")
        # display the updated menu items
        screen.blit(start, (100,220))
        screen.blit(quit, (100,290))
        pygame.display.update()
        for event in pygame.event.get():
            #if the close button is pressed
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == KEYDOWN:
                #if the escape key is pressed
                if event.key == K_ESCAPE:
                    sys.exit()
                #if the down button is pressed
                elif event.key == K_DOWN:
                    if selected == 0:
                        selected = 1
                #if the up button is pressed
                elif event.key == K_UP:
                    if selected == 1:
                        selected = 0
                #if the enter (return) button is pressed
                elif event.key == K_RETURN:
                    if selected == 0:
                        displayMenu = False
                    elif selected == 1:
                        sys.exit()



# Load the level from the text file
def LoadLevel(name, screen, music, oldsky=("error.png")):
    # These are the sprite groups
    staticG = pygame.sprite.Group()
    # movingG is all the objects that require constant updating
    movingG = pygame.sprite.Group()
    octopusG = pygame.sprite.Group()
    backgroundG = pygame.sprite.Group()
    colliderG = pygame.sprite.Group()
    deathG = pygame.sprite.Group()
    skyG = pygame.sprite.Group()
    sounds = []
    finish = []
    # This adds the Class to the group
    Dirt.containers = staticG
    Stone.containers = staticG
    Grass.containers = staticG
    Waves.containers = staticG
    Launcher.containers = movingG
    Rocket.containers = movingG
    Ship.containers = movingG
    sky = oldsky
    Collider.containers = colliderG
    Animation.containers = movingG
    TP.containers = movingG
    Insert.containers = movingG
    Package.containers = movingG
    Water.containers = staticG
    Water.containers = staticG
    Bullet.containers = deathG
    Sphere.containers = deathG
    Bomb.containers = deathG
    Boss.containers = deathG
    Explosion.containers = deathG
    Octopus.containers = octopusG
    # is this level a cutscene
    cut = False
    file = data.load(name).readlines()
    for line in file:
        part = line.split(",")
        if part[0] == "dirt":
            Dirt(int(part[1]), int(part[2]))
        elif part[0] == "stone":
            Stone(int(part[1]), int(part[2]))
        elif part[0] == "grass":
            Grass(int(part[1]), int(part[2]))
        elif part[0] == "sky":
            sky.update(part[1])
        elif part[0] == "water":
            Water(int(part[1]), int(part[2]))
        elif part[0] == "waves":
            Waves(int(part[1]), int(part[2]))
        elif part[0] == "tp":
            TP(int(part[1]), int(part[2]))
        elif part[0] == "sphere":
            Sphere(int(part[1]), int(part[2]))
        elif part[0] == "launcher":
            Launcher(int(part[1]), int(part[2]), int(part[3]))
        elif part[0] == "animation":
            Animation(int(part[1]), int(part[2]), int(part[3]), int(part[4]), int(part[5]), part[6])
        elif part[0] == "ship":
            Ship(int(part[1]), int(part[2]), int(part[3]))
        elif part[0] == "boss":
            Boss(int(part[1]), int(part[2]))
        elif part[0] == "cut":
            cut = True
            Insert(200, 32, 1, 300, "Inserts/skip.png")
        elif part[0] == "insert":
            Insert(int(part[1]), int(part[2]), int(part[3]), int(part[4]), part[5])
        elif part[0] == "sound":
            sounds.append(PlaySound(int(part[1]), int(part[2]), int(part[3]), int(part[4]), part[5]))
        elif part[0] == "music":
            music.newSong(part[1])
        elif part[0] == "end":
            finish.append(Finish(int(part[1])))
        elif part[0] == "octopus":
            Octopus(int(part[1]), int(part[2]))
            Collider((int(part[1]), int(part[2])), (1, 30), (42, 1), "down")
            Collider((int(part[1]), int(part[2])), (-1, 8), (1, 22), "left")
            Collider((int(part[1]), int(part[2])), (44, 8), (1, 22), "right")
            Collider((int(part[1]), int(part[2])), (1, 7), (42, 1), "up")

    screen.fill(BLUE)
    skyG.draw(screen)
    backgroundG.draw(screen)
    staticG.draw(screen)
    deathG.draw(screen)
    movingG.draw(screen)
    octopusG.draw(screen)
    colliderG.draw(screen)
    pygame.display.update()
    #time.sleep(0.1)
    return backgroundG, staticG, octopusG, colliderG, movingG, deathG, sounds, finish, music, sky, cut


import base64
import dill
import numpy as np
from pygame.event import Event

try:
    from pyrl.tasks.pygame.utils import AsyncEvent, SyncEvent
    print 'succeeded in importing pygame utils'
except ImportError:
    raise ImportError('failed to import pygame utils')


GO_FLAG = [0] # how many times to update the screen before asking for next action.

# the actual game code
def Game(screen, event_gen = AsyncEvent(), level = 0):
    # start game level.
    music = PlayMusic(None)
    skys = pygame.sprite.Group()
    Sky.containers = skys
    gameOver = False
    sky=Sky("Sky.png")
    if gameOver == False:
        levelOver = False
        levelRestart = [False]
        if type(level) == int:
            level += 1
            level_path = os.path.join("Levels", str(level) + ".txt")
        else:
            level_path = os.path.join("Levels/", level + ".txt")
        print 'level_path', level_path

        pygame.display.set_caption('Mr Octopus!!! Level ' + str(level))
        Sky.remove(sky)
        try:
            backgroundG, staticG, octopusG, colliderG, movingG, deathG, sounds, finish, music, sky, cut = LoadLevel(level_path, screen, music=music, oldsky=sky)
            up = down = left = right = False
            water = False
            jumped = False
        except:
            levelOver = True
            gameOver = True

        # mount handlers.
        def get_state_dict():
            state_dict = {}
            for group in [staticG, octopusG, movingG]:
                for sprite in group.sprites():
                    if sprite.type not in state_dict:
                        state_dict[sprite.type] = []
                    state_dict[sprite.type].append((sprite.x, sprite.x+sprite.rect.w, sprite.y, sprite.y+sprite.rect.h))
            return state_dict

        def get_gameover():
            if levelOver:
                levelRestart[0] = True
            return levelOver

        def ready_to_go():
            GO_FLAG[0] = 100


        event_gen.mount('state_dict', get_state_dict)
        event_gen.mount('gameover', get_gameover)
        event_gen.mount('go', ready_to_go)

        while not levelRestart[0]:
            #this handles all the events
            for event in event_gen.get():
                print 'event', event
                if event.type == QUIT:
                    sys.exit()
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        Menu(screen)
                    # here is all the keypresses for the movement
                    if event.key == K_DOWN:
                        down = True
                    elif event.key == K_UP:
                        up = True
                    elif event.key == K_LEFT:
                        left = True
                    elif event.key == K_RIGHT:
                        right = True
                    elif event.key == K_r:
                        if levelOver == False:
                            levelOver = True
                            level -= 1
                    elif event.key == K_s and cut == True:
                        levelOver = True
                        for item in sounds:
                            if item.playing == True:
                                item.sound.stop(1)
                elif event.type == KEYUP:
                    if event.key == K_DOWN:
                        down = False
                    elif event.key == K_UP:
                        up = False
                    elif event.key == K_LEFT:
                        left = False
                    elif event.key == K_RIGHT:
                        right = False

            if GO_FLAG[0] == 0: # do not simulate, just update screen to avoid blocking.
                pygame.display.update()
                #continue
            else:
                GO_FLAG[0] -= 1


            for item in octopusG:
                water = False
                # print item.x, ',', item.y
                # slow down in water
                for staticitem in staticG:
                    if item.rect.colliderect(staticitem.rect):
                        if staticitem.type == "water" or staticitem.type == "waves":
                            item.fallspeed /= 1.2
                            water = True
                # kill on enemies
                for deathitem in deathG:
                    if item.rect.colliderect(deathitem.rect):
                        if levelOver == False:
                            levelOver = True
                            level -= 1
                direction = 0
                if down == True and item.down == True and water == True:
                    item.y += 1.5
                if up == True:
                    if (item.down == False or water == True) and (item.up == True):
                        item.y -= 3.2
                        # to test if the character was jumping last frames
                        jumped = True
                    elif jumped == True and item.up == True:
                        item.y -= 3.2
                    else:
                        jumped = False
                else:
                    if jumped == True:
                        item.fallspeed -= 3
                    jumped = False
                if left == True and item.left == True:
                    item.x -= 2
                    direction= 1
                if right == True and item.right == True:
                    item.x += 2
                    direction = -1
                item.update(direction)
                # this is moving the boundary boxex to where the character is
                for collideritem in colliderG:
                    collideritem.x = item.x
                    collideritem.y = item.y
                    collideritem.update()
                # this is so that later on I can make it false if the octopus is colliding with an object
                # i do it so that the character doesn't walk through walls
                item.down = True
                item.left = True
                item.right = True
                item.up = True

            for item in colliderG:
                for staticitem in staticG:
                    if item.rect.colliderect(staticitem.rect):
                        if staticitem.type == "dirt" or staticitem.type == "grass" or staticitem.type == "stone":
                            if item.type == "down":
                                for octopus in octopusG:
                                    octopus.fallspeed = 0
                                    octopus.down = False
                                    octopus.y = staticitem.rect.top - octopus.rect.h
                            if item.type == "up":
                                for octopus in octopusG:
                                    octopus.y = staticitem.rect.y + staticitem.rect.h - item.oy
                                    octopus.fallspeed = 0
                                    octopus.up = False
                            if item.type == "left":
                                for octopus in octopusG:
                                    octopus.left = False
                            if item.type == "right":
                                for octopus in octopusG:
                                    octopus.right = False

                for movingitem in movingG:
                    if item.rect.colliderect(movingitem.rect):
                        if movingitem.type == "teleporter":
                            levelOver = True


            for item in movingG:
                item.update()
                if item.type == "package":
                    for staticitem in staticG:
                        if staticitem.type == "dirt" or staticitem.type == "grass" or staticitem.type == "stone":
                            if item.rect.colliderect(staticitem.rect):
                                item.falling = False
                                item.y = staticitem.rect.top - item.rect.h
                    for octopus in octopusG:
                        if item.rect.colliderect(octopus.rect):
                            Rocket(item.x, item.y)
                            movingG.remove(item)
                if item.type == "ship":
                    if item.shoot == True:
                        Bomb(item.rect.centerx, item.rect.centery, item.direction)
                    for staticitem in staticG:
                        if staticitem.type == "dirt" or staticitem.type == "grass" or staticitem.type == "stone":
                            if item.rect.colliderect(staticitem.rect):
                                item.direction = -item.direction
                                item.x += item.direction
                                item.location += item.direction
                if item.type == "launcher":
                    if item.shoot == True:
                        Bullet(item.x, item.y)

            for item in sounds:
                item.update()

            for item in deathG:
                item.update()
                if item.type == "boss":
                    if item.shoot == True:
                        Bomb(item.rect.centerx + random.randint(-40, 40), item.rect.centery, random.randint(-8, 8))
                    if item.shoot2 == True:
                        Package(item.rect.centerx + random.randint(-40, 40), item.rect.centery, random.randint(-1, 1))
                    for staticitem in staticG:
                        if staticitem.type == "dirt" or staticitem.type == "grass" or staticitem.type == "stone":
                            if item.rect.colliderect(staticitem.rect):
                                item.direction = -item.direction
                                item.x += item.direction
                                item.location += item.direction
                    for movingitem in movingG:
                        if movingitem.type == "rocket":
                            if item.rect.colliderect(movingitem.rect):
                                item.health -= 1
                                Explosion(movingitem.x, movingitem.y)
                                movingG.remove(movingitem)
                                if item.health == 0:
                                    deathG.remove(item)
                                    finish.append(Finish(1))
                elif item.type == "bullet":
                    if item.x < -100:
                        deathG.remove(item)
                elif item.type == "bomb":
                    for staticitem in staticG:
                        if staticitem.type == "dirt" or staticitem.type == "grass" or staticitem.type == "stone":
                            if item.rect.colliderect(staticitem.rect):
                                Explosion(item.x, item.y)
                                deathG.remove(item)
                elif item.type == "explosion":
                    if item.location > 3:
                        deathG.remove(item)

            for item in finish:
                item.update()
                if item.finished == True:
                    levelOver = True

            skys.draw(screen)
            backgroundG.draw(screen)
            staticG.draw(screen)
            #deathG.draw(screen)
            movingG.draw(screen)
            octopusG.draw(screen)
            #colliderG.draw(screen)
            pygame.display.update()


def setup_screen():
    # this sets up the screen
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
    pygame.display.set_caption('Mr Octopus!!!')
    pygame.display.set_icon(loadIm("icon.png"))
    screen.fill(BLUE)
    return screen


def start_game(event=AsyncEvent(), level=0):
    pygame.init()
    screen = setup_screen()

    ##while True:
    #if 'video' in os.environ:
    #    video_fname = os.environ['video']
    #    from pyrl.tasks.pygame.utils import VideoRecorder
    #    recorder = VideoRecorder(screen, video_fname)

    Game(screen, event, level)


