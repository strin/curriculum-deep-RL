# London's burning
# a game by Adam Binks

import pygame

BASICFONT = pygame.font.Font('assets/fonts/arcadeclassic.ttf', 24) # http://www.1001fonts.com/arcadeclassic-font.html
BIGFONT = pygame.font.Font('assets/fonts/arcadeclassic.ttf', 36) # http://www.1001fonts.com/arcadeclassic-font.html
MEGAFONT = pygame.font.Font('assets/fonts/arcadeclassic.ttf', 52) # http://www.1001fonts.com/arcadeclassic-font.html

def genText(text, topLeftPos, colour, font):
	surf = font.render(text, 1, colour)
	surf.set_colorkey((0,0,0))
	rect = surf.get_rect()
	rect.topleft = topLeftPos
	return (surf, rect)
