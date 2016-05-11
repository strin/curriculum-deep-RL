# London's burning
# a game by Adam Binks
import pygame
pygame.mixer.pre_init(44100, -16, 2, 512)   # use a lower buffersize to reduce sound latency
pygame.init()

import input, game, random, sound, ui, webbrowser

def run():
	stateHandler = StateHandler()
	while True:
		stateHandler.update()



class StateHandler:
	"""handles menu and game state, runs boilerplate update code"""
	def __init__(self):
		self.data = Data()
		self.data.screen.fill(self.data.grey)
		pygame.display.set_caption('London\'s Burning!')

		pygame.mouse.set_cursor(*pygame.cursors.diamond)

		self.mainMenu = MainMenu(self.data)
		self.gameHandler = None

		sound.playMusic('assets/sounds/searching.mp3')   # courtesy of http://ericskiff.com/music/
														 # released under CC Attribution license


	def update(self):
		self.data.input.get(self.data, (self.gameHandler is not None and not self.gameHandler.gameOver))
		self.data.dt = self.data.FPSClock.tick(self.data.FPS) / 100.0

		# update game/menu objs
		if self.gameHandler:
			self.gameHandler.update(self.data, self.data.dt)
		
		else:
			done = self.mainMenu.update(self.data)
			if done:
				self.gameHandler = game.GameHandler(self.data)


		pygame.display.update()



class Data:
	"""stores variables to be accessed in many parts of the game"""
	def __init__(self):
		self.WINDOWWIDTH, self.WINDOWHEIGHT = (600, 800)
		input.Input.winSize = [self.WINDOWWIDTH, self.WINDOWHEIGHT]

		self.screen = pygame.display.set_mode((self.WINDOWWIDTH, self.WINDOWHEIGHT))
		self.FPSClock = pygame.time.Clock()
		self.FPS = 60
		self.input = input.Input()
		sound.muted = False

		self.IMAGESCALEUP = 4

		self.screenShakeOffset = [0, 0]

		self.grey = (60, 60, 60)
		self.white = (250, 250, 250)
		self.lightgrey = (170, 170, 170)
		self.yellow = (255, 223, 134)

		self.scoreValues = {'shoot bomb': 20, 'shoot superBomb': 30, 'destroy bomber': 200}


	def newGame(self):
		self.score = 0

		self.gameSurf = pygame.Surface((self.WINDOWWIDTH, self.WINDOWHEIGHT))
		self.gameSurf.convert()

		self.destroyableEntities = pygame.sprite.Group() # any objects destroyable by bombs
		self.bomberTargetedEntities = pygame.sprite.Group()  # any objects that bombers will try and drop bombs on
		self.bulletproofEntities = pygame.sprite.Group() # bullets bounce straight of these
		self.superbombableEntities = pygame.sprite.Group() # any objects destroyed by floating super bombs

		self.particles = pygame.sprite.Group()
		self.particleSpawners = pygame.sprite.Group()

		self.buildings = pygame.sprite.Group()
		self.standingBuildings = pygame.sprite.Group()

		self.bombers = pygame.sprite.Group()
		self.bombs = pygame.sprite.Group()
		self.superBombs = pygame.sprite.Group()

		self.AAguns = pygame.sprite.Group()
		self.bullets = pygame.sprite.Group()

		self.spotlights = pygame.sprite.Group()


	def shakeScreen(self, intensity):
		"""Gives a screenshake effect - for the next few frames the gameSurf will be blitted at an offset"""
		for axis in (0, 1):
			self.screenShakeOffset[axis] += random.choice([-1.0 * intensity, 1.0 * intensity])


	def updateScreenshake(self):
		"""Begins to return the screenShakeOffset to [0, 0] frame by frame"""
		for axis in (0, 1):
			self.screenShakeOffset[axis] -= self.screenShakeOffset[axis] * 0.5 * self.dt
			if random.randint(0, 10) == 0:
				self.screenShakeOffset[axis] = -self.screenShakeOffset[axis]


	def loadImage(self, imagePath, customScaleFactor=None):
		"""Loads an image and scales it 4x so it's nice and pixellated"""
		if customScaleFactor: scaleFactor = customScaleFactor
		else: scaleFactor = self.IMAGESCALEUP

		img = pygame.image.load(imagePath)
		img = pygame.transform.scale(img, (img.get_width() * scaleFactor, img.get_height() * scaleFactor))
		img.convert_alpha()
		return img


	def saveGame(self):
		pass



class MainMenu:
	def __init__(self, data):
		self.surfs = []
		self.rects = []

		surf, rect = ui.genText('Londons Burning!', (0, 0), data.yellow, ui.MEGAFONT)
		rect.midbottom = (data.WINDOWWIDTH / 2, data.WINDOWHEIGHT / 2 - 50)
		self.surfs.append(surf)
		self.rects.append(rect)

		texts = ['Defend  London  from  death  from  above!',
				 'Click  to  shoot  the  bombs  out  of  the  air',
				 'You  cant  directly  shoot  the  bombers',
				 'Shoot  the  big  bombs  to  ricochet  them  back',
				 'and  blow  up  the  bombers!',
				 'Press  any  key  to  continue']

		y = self.rects[0].bottom + 50
		for text in texts:
			surf, rect = ui.genText(text, (0, 0), data.lightgrey, ui.BASICFONT)

			if text == 'Press  any  key  to  continue':
				surf, rect = ui.genText(text, (0, 0), data.white, ui.BASICFONT)

			rect.midbottom = (data.WINDOWWIDTH / 2, y)
			self.surfs.append(surf)
			self.rects.append(rect)
			y += rect.height + 20


		textSurf1, textRect1 = ui.genText('music by eric skiff', (0, 0), data.lightgrey, ui.BASICFONT)
		textRect1.bottomright = (data.WINDOWWIDTH - 10, data.WINDOWHEIGHT - 10)
		self.surfs.append(textSurf1)
		self.rects.append(textRect1)
		self.musicCreditsRect = textRect1

		textSurf2, textRect2 = ui.genText('created by adam binks', (0, 0), data.yellow, ui.BASICFONT)
		textRect2.bottomright = (data.WINDOWWIDTH - 10, textRect1.top - 5)
		self.surfs.append(textSurf2)
		self.rects.append(textRect2)

		logoSurf = data.loadImage('assets/ui/jellyberg.png')
		logoRect = logoSurf.get_rect(bottomright = (data.WINDOWWIDTH - 10, textRect2.top - 5))
		self.surfs.append(logoSurf)
		self.rects.append(logoRect)
		self.jellybergCreditsRects = [textRect2, logoRect]

		bombImg = data.loadImage('assets/enemies/superBomb.png', 14)
		bombRect = bombImg.get_rect(midbottom = (data.WINDOWWIDTH / 2, self.rects[0].top - 50))
		self.surfs.append(bombImg)
		self.rects.append(bombRect)


		self.musicSurf = {}
		self.musicSurf[1] = data.loadImage('assets/ui/musicON.png')
		self.musicSurf[0] = data.loadImage('assets/ui/musicOFF.png')
		self.musicRect = self.musicSurf[0].get_rect(topright = (data.WINDOWWIDTH - 10, 10))
		self.musicMuted = False

		self.sfxSurf = {}
		self.sfxSurf[1] = data.loadImage('assets/ui/sfxON.png')
		self.sfxSurf[0] = data.loadImage('assets/ui/sfxOFF.png')
		self.sfxRect = self.sfxSurf[0].get_rect(topright =  (self.musicRect.left - 5, 10))


	def update(self, data):
		data.screen.fill(data.grey)

		for i in range(len(self.surfs)):
			data.screen.blit(self.surfs[i], self.rects[i])

		if data.input.mouseUnpressed == 1:
			# if logo or my name is clicked, go to my itch.io page
			for rect in self.jellybergCreditsRects:
				if rect.collidepoint(data.input.mousePos):
					webbrowser.open('http://jellyberg.itch.io/')

			if self.musicCreditsRect.collidepoint(data.input.mousePos):
				webbrowser.open('http://ericskiff.com/music/')

			if self.sfxRect.collidepoint(data.input.mousePos):
				sound.muted = not sound.muted
			elif self.musicRect.collidepoint(data.input.mousePos):
				self.musicMuted = not self.musicMuted
				if self.musicMuted:
					sound.pauseMusic()
				else:
					sound.resumeMusic()

		data.screen.blit(self.sfxSurf[not sound.muted], self.sfxRect)
		data.screen.blit(self.musicSurf[not self.musicMuted], self.musicRect)


		if data.input.unpressedKeys:  # start the game
			return 'done'



if __name__ == '__main__':
	run()