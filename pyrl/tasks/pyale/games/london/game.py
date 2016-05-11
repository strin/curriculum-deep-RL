# London's burning
# a game by Adam Binks

import ui, random, time
from object import Building, Bomber, AAGun, Bomb, SuperBomb, Bullet, Spotlight

class GameHandler:
	def __init__(self, data):
		Bomb.baseImage = data.loadImage('assets/enemies/bomb.png')
		SuperBomb.baseImage = data.loadImage('assets/enemies/superBomb.png')
		Bullet.image = data.loadImage('assets/defences/bullet.png')
		Spotlight.baseImage = data.loadImage('assets/defences/spotlight.png')
		Spotlight.standImage = data.loadImage('assets/defences/spotlightStand.png')

		self.newGame(data)

		self.lastEndScore = 0 # score achieved in the last game
		self.highScore = 0 # best score yet

		# starting difficulty values
		self.minBomberInterval = 13.0
		self.maxBomberInterval = 18.0
		self.learningCurve = 0.5  # how much the spawn rate of bombers increases per bomber spawn
		self.lowestEverBomberSpawnInterval = 1 # min bomber spawn interval never goes below this


	def update(self, data, dt):
		data.gameSurf.fill(data.grey)
		data.screen.fill(data.grey)

		data.spotlights.update(data)
		data.buildings.update(data)
		
		data.particleSpawners.update(data)
		data.particles.update(data)

		if not self.gameOver:
			data.AAguns.update(data)
		data.bullets.update(data)

		if time.time() - self.lastBomberTime > self.timeTillNewBomber or len(data.bombers) == 0:
			Bomber(data, random.randint(0, 1), random.randint(50, 250))
			self.timeTillNewBomber = random.uniform(self.minBomberInterval, self.maxBomberInterval)
			self.lastBomberTime = time.time()

			if self.minBomberInterval > self.lowestEverBomberSpawnInterval:
				self.minBomberInterval -= self.learningCurve
				self.maxBomberInterval -= self.learningCurve
		data.bombers.update(data)
		data.bombs.update(data)

		data.screen.blit(data.gameSurf, (data.screenShakeOffset[0], data.screenShakeOffset[1]))
		data.updateScreenshake()

		# DISPLAY SCORE
		if not self.gameOver:
			if data.score != self.lastScore:
				self.scoreSurf, self.scoreRect = ui.genText('SCORE   %s' %(data.score), (0, 0), data.white, ui.BASICFONT)
				self.scoreRect.topright = (data.WINDOWWIDTH - 10, 10)

			data.screen.blit(self.scoreSurf, self.scoreRect)
			self.lastScore = data.score

		if not self.gameOver:
			self.checkForGameOver(data)

		if self.gameOver:
			self.showGameOverOverlay(data)

			if len(data.input.unpressedKeys) > 0:# start a new game if a or mouse button key is pressed
				self.gameOver = False
				self.newGame(data)


	def newGame(self, data):
		data.newGame()

		self.gameOver = False
		self.lastScore = -1 # score last frame

		Spotlight(data, (200, data.WINDOWHEIGHT), 50)
		Spotlight(data, (400, data.WINDOWHEIGHT), 130)

		self.timeTillNewBomber = random.randint(14, 16) # longer time till second bomber to get into the swing of things
		self.lastBomberTime = time.time()
		Bomber(data, random.randint(0, 1), random.randint(50, 300))

		buildingImgs = []
		for buildingFileName in ['tall1', 'tall2', 'small1', 'small2', 'small3', 'factory1']:
			buildingImgs.append(data.loadImage('assets/buildings/%s.png' %(buildingFileName)))
		aaGunIsBuilt = False
		x = 4
		while x < data.WINDOWWIDTH:
			img = random.choice(buildingImgs)
			if x > int(data.WINDOWWIDTH / 2) - 32 and not aaGunIsBuilt: # build a single AA Gun roughly halfway across the screen
				AAGun(data, x)
				x += 64
				aaGunIsBuilt = True

			elif x + img.get_width() < data.WINDOWWIDTH:
				Building(data, img, x)
				x += img.get_width()
			x += 4 # gap between buildings


	def checkForGameOver(self, data):
		"""Check if all buildings or the AA gun have been bombed"""
		if len(data.standingBuildings) == 0 or len(data.AAguns) == 0:
			self.gameOver = True
			self.lastEndScore = data.score
			if self.highScore < data.score:
				self.highScore = data.score
			self.genGameOverOverlay(data)


	def genGameOverOverlay(self, data):
		"""Generates the objects neccessary for the game over screen"""
		self.gameOverStuffSurfs = []
		self.gameOverStuffRects = []

		surf, rect = ui.genText('GAME OVER', (0, 0), data.yellow, ui.MEGAFONT)
		rect.midbottom = (data.WINDOWWIDTH / 2, data.WINDOWHEIGHT / 2 - 20)
		self.gameOverStuffSurfs.append(surf)
		self.gameOverStuffRects.append(rect)

		surf, rect = ui.genText('score   %s' %(self.lastEndScore), (0, 0), data.white, ui.BASICFONT)
		rect.bottomright = (data.WINDOWWIDTH / 2 - 10, data.WINDOWHEIGHT / 2)
		self.gameOverStuffSurfs.append(surf)
		self.gameOverStuffRects.append(rect)

		surf, rect = ui.genText('high score   %s' %(self.highScore), (0, 0), data.white, ui.BASICFONT)
		rect.bottomleft = (data.WINDOWWIDTH / 2 + 10, data.WINDOWHEIGHT / 2)
		self.gameOverStuffSurfs.append(surf)
		self.gameOverStuffRects.append(rect)

		surf, rect = ui.genText('press any key to continue', (0, 0), data.lightgrey, ui.BASICFONT)
		rect.midbottom = (data.WINDOWWIDTH / 2 + 10, data.WINDOWHEIGHT / 2 + 50)
		self.gameOverStuffSurfs.append(surf)
		self.gameOverStuffRects.append(rect)


	def showGameOverOverlay(self, data):
		"""Shows a game over screen on top of the bombers buzzing about"""
		for i in range(len(self.gameOverStuffSurfs)):
			data.screen.blit(self.gameOverStuffSurfs[i], self.gameOverStuffRects[i])

