# London's burning
# a game by Adam Binks

import pygame, random

'''
SOUND = {}
for filename in ['explosion1', 'explosion2', 'explosion3', 'explosion4',
				 'explosion5', 'hit', 'plup', 'shoot1', 'shoot2', 'pip',
				 'swoosh', 'shoot down']: # .wav files only
	SOUND[filename] = pygame.mixer.Sound('assets/sounds/%s.wav' %(filename))
'''

def play(sound, volume=0.8, varyVolume=True, loops=0):
        return
	"""Plays the given sound"""
	if muted: return

	if varyVolume:
		volume -= random.uniform(0.0, 0.2)
		if volume < 0.1: volume == 0.1
		SOUND[sound].set_volume(volume)
	SOUND[sound].play(loops)


def playMusic(filename):
    return
    if filename.rfind('.mp3') != -1:
        return
	pygame.mixer.music.load(filename)
	pygame.mixer.music.set_volume(0.5)
	pygame.mixer.music.play(-1)


def pauseMusic():
    return
    pygame.mixer.music.pause()


def resumeMusic():
    return
    pygame.mixer.music.play(-1)
