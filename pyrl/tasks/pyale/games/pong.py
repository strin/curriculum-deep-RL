#
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation; either version 2 of the License, or
#       (at your option) any later version.
#
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#
#		It's my first actual game-making attempt. I know code could be much better
#		with classes or defs but I tried to make it short and understandable with very
#		little knowledge of python and pygame(I'm one of them). Enjoy.
import pygame
import numpy as np
import numpy.random as npr
from pygame.locals import *
import sys, os
import random

pygame.init()
pygame.font.init()
pygame.display.set_caption("Pong Pong!")

screen=pygame.display.set_mode((640,480),0,32)

W = 20
try:
    H1 = int(os.environ['H1'])
except:
    H1 = 50
try:
    H2 = int(os.environ['H2'])
except:
    H2 = 50
C = 15
show_score = False

#Creating 2 bars, a ball and background.
height = 480
width = 640
back = pygame.Surface((width,height))
background = back.convert()
background.fill((144,72,17))
bar1 = pygame.Surface((W,H1)).convert()
bar1.fill((101, 213, 77))
bar2 = pygame.Surface((W,H2)).convert()
bar2.fill((213, 130, 74))
circ_sur = pygame.Surface((C,C))
# circ = pygame.draw.circle(circ_sur,(0,255,0),(C/2,C/2),C/2) # circle shape
circ = circ_sur # rect shape
circ.fill((255, 255, 255)) 
circle = circ_sur.convert()
circle.set_colorkey((0,0,0))


# some definitions
bar1_x, bar2_x = 2 * W , 640 - 3 * W
bar1_y, bar2_y = 215. , 215.
init_circle_y = lambda: (640 - 60) * npr.rand() + 30
circle_x, circle_y = 320.5, init_circle_y()
bar1_move, bar2_move = 0. , 0.
speed_x, speed_y, speed_circ, = 250., 250., 250.
speed_me = 250.
speed_ai = 250. # adjust this parameter to set AI strength.
base_speed_y = speed_y
base_speed_x = speed_x
speed_y = np.sign(npr.randn()) * speed_y # randomize direction.
ai_speed = 0.
me_speed = 0.
bar1_score, bar2_score = 0,0

#clock and font objects
clock = pygame.time.Clock()

if show_score:
    font = pygame.font.SysFont("Eurostile", 70)

get_event = lambda: pygame.event.get()
is_end = False
opening = 1

while not is_end:
    for event in get_event():
        if event.type == QUIT:
            is_end = True # TODO: figure out how to intercept exit.
            break
        if event.type == KEYDOWN:
            if event.key == K_UP:
                bar1_move = -me_speed
            elif event.key == K_DOWN:
                bar1_move = me_speed
        elif event.type == KEYUP:
            if event.key == K_UP:
                bar1_move = 0.
            elif event.key == K_DOWN:
                bar1_move = 0.


    screen.blit(background,(0,0))
    #frame = pygame.draw.rect(screen,(255,255,255),Rect((5,5),(630,470)),2)
    #middle_line = pygame.draw.aaline(screen,(255,255,255),(330,5),(330,475))
    screen.blit(bar1,(bar1_x,bar1_y))
    screen.blit(bar2,(bar2_x,bar2_y))
    screen.blit(circle,(circle_x,circle_y))

    if show_score:
        score1 = font.render(str(bar1_score), True, (101, 213, 77))
        score2 = font.render(str(bar2_score), True, (213, 130, 74))
        screen.blit(score1,(200.,50.))
        screen.blit(score2,(430.,50.))

    bar1_y += bar1_move

    # movement of circle
    #clock.tick(30)
    time_passed = 30
    time_sec = time_passed / 1000.0

    circle_x += speed_x * time_sec
    circle_y += speed_y * time_sec
    ai_speed = speed_ai * time_sec
    me_speed = speed_me * time_sec
#AI of the computer.
    if circle_x >= 320 - C / 2.:
        if npr.rand() < 0.: # random action.
            bar2_y += np.sign(npr.randn()) * ai_speed
        else:
            noise = 15 * (1 - opening) * npr.randn()
            if not bar2_y == circle_y + C / 2.:
                if bar2_y + H2 / 2. + noise < circle_y + C / 2.:
                    bar2_y += ai_speed
                if  bar2_y + H2/ 2.  + noise > circle_y  + C / 2.:
                    bar2_y -= ai_speed
            else:
                bar2_y == circle_y + C / 2.

    if bar1_y >= 480. - H1: bar1_y = 480. - H1
    elif bar1_y <= 10. : bar1_y = 10.
    if bar2_y >= 480. - H2: bar2_y = 480 - H2
    elif bar2_y <= 10.: bar2_y = 10.
#since i don't know anything about collision, ball hitting bars goes like this.
    if circle_x <= bar1_x + W and circle_x >= bar1_x + W - C:
        if circle_y >= bar1_y - C / 2. and circle_y <= bar1_y + H1 - C / 2.:
            circle_x = bar1_x + W
            speed_x = base_speed_x + npr.randn() * 5
            speed_y = speed_y + abs(npr.randn()) * 5 * (np.sign(bar1_move))
            opening = 0
    if circle_x >= bar2_x - C and circle_x <= bar2_x:
        if circle_y >= bar2_y - C / 2. and circle_y <= bar2_y + H2 - C / 2.:
            circle_x = bar2_x - C
            speed_x = -base_speed_x + npr.randn() * 5
            speed_y = speed_y + abs(npr.randn()) * 5 * (np.sign(ai_speed))
            opening = 0
    if circle_x < 5.:
        bar2_score += 1
        circle_x, circle_y = 307.5, init_circle_y()
        speed_y = base_speed_y * np.sign(npr.randn())
        speed_x = abs(base_speed_x)
        bar1_y, bar2_y = 215., 215.
        opening = 1
    elif circle_x > 620.:
        bar1_score += 1
        circle_x, circle_y = 307.5, init_circle_y() 
        speed_y = base_speed_y * np.sign(npr.randn())
        speed_x = abs(base_speed_x)
        bar1_y, bar2_y = 215., 215.
        opening = 1
    if circle_y <= 10.:
        speed_y = -speed_y
        circle_y = 10.
    elif circle_y >= 457.5:
        speed_y = -speed_y
        circle_y = 457.5

    pygame.display.update()


