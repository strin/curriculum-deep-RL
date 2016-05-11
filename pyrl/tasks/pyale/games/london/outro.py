import pygame, input, webbrowser

BASICFONT = pygame.font.Font('assets/fonts/arcadeclassic.ttf', 24) # http://www.1001fonts.com/arcadeclassic-font.html
BIGFONT = pygame.font.Font('assets/fonts/arcadeclassic.ttf', 36) # http://www.1001fonts.com/arcadeclassic-font.html
MEGAFONT = pygame.font.Font('assets/fonts/arcadeclassic.ttf', 52) # http://www.1001fonts.com/arcadeclassic-font.html

def genText(text, topLeftPos, colour, font):
	surf = font.render(text, 1, colour)
	surf.set_colorkey((0,0,0))
	rect = surf.get_rect()
	rect.topleft = topLeftPos
	return (surf, rect)

class ThanksForPlaying:
	"""Popup that thanks the player and shows a couple of links to other stuff"""
	top = 150  # distance from top of window
	def __init__(self, data):
		self.buttons = []
		texts = ['Resume', 'Play  my  other  games!', 'Read  my  devblog!', 
					 'Hear  about  my  next  game!', 'Exit']
		for i in range(len(texts)):
			self.buttons.append({'text': texts[i]})
			self.buttons[i]['surf'], self.buttons[i]['rect'] = genText(texts[i], (0, 0), (255, 255, 255), BIGFONT)
			self.buttons[i]['rect'].midbottom = (data.WINDOWWIDTH / 2, ThanksForPlaying.top + i*40)

		self.greySurf = pygame.Surface((data.WINDOWWIDTH, data.WINDOWHEIGHT))
		self.greySurf.fill((0, 0, 0))
		self.greySurf.set_alpha(150)

		self.prevScreen = data.screen.copy()
		self.prevScreen.blit(self.greySurf, (0, 0))


	def update(self, data):
		data.input.get(data, False, True)

		data.screen.blit(self.prevScreen, (0, 0))

		for button in self.buttons:
			data.screen.blit(button['surf'], button['rect'])

			if button['rect'].collidepoint(data.input.mousePos) and data.input.mouseUnpressed == 1:
				if button['text'] == 'Resume':
					return 'DONE'
				if 'games' in button['text']:
					webbrowser.open('http://jellyberg.itch.io/')
				if 'devblog' in button['text']:
					webbrowser.open('http://jellybergfish.tumblr.com')
				if 'next  game' in button['text']:
					webbrowser.open('http://jellybergfish.tumblr.com/mailing-list')
				if button['text'] == 'Exit':
					data.input.terminate()
		pygame.display.update()


def showOutro(data):
	outro = ThanksForPlaying(data)
	done = False

	while not done:
		done = outro.update(data)
	data.FPSClock.tick(data.FPS)
