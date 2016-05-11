# London's burning
# a game by Adam Binks

import pygame, time, random, sound

class Particle(pygame.sprite.Sprite):
	"""A simple particle"""
	gravity = 1.5
	def __init__(self, data, image, topleft, avgLifeSpan, velocity, obeysGravity):
		"""obeysGravity: 0 is no gravity, 1 is falling particle, -1 is upward floating particle"""
		pygame.sprite.Sprite.__init__(self)
		self.add(data.particles)

		self.image = image
		self.rect = image.get_rect(topleft=topleft)
		self.coords = list(topleft)

		self.velocity = list(velocity)
		self.gravity = obeysGravity

		self.birthTime = time.time()
		self.lifeTime = random.uniform(avgLifeSpan - 0.3, avgLifeSpan + 0.3)
		if self.lifeTime < 0: self.lifeTime = 0.05


	def update(self, data):
		if self.gravity != 0:
			self.velocity[1] = self.velocity[1] + Particle.gravity * self.gravity * data.dt # allow gravity to act upon the particle
		self.coords[0] += self.velocity[0] * data.dt
		self.coords[1] += self.velocity[1] * data.dt

		self.rect.topleft = self.coords

		data.gameSurf.blit(self.image, self.rect)
		if time.time() - self.birthTime > self.lifeTime:
			self.kill()



class SmokeSpawner(pygame.sprite.Sprite):
	"""A particle spawner for smoke"""
	def __init__(self, data, topleft, intensity, lifespan=None):
		pygame.sprite.Sprite.__init__(self)
		self.add(data.particleSpawners)

		self.sourceCoords = topleft
		smokeImage = data.loadImage('assets/particles/smoke.png')
		self.particleLifespan = 0.2
		self.intensity = intensity
		self.lifespan = lifespan
		self.birthTime = time.time()

		self.imgs = []
		for degrees in [0, 90, 180, 270]:
			self.imgs.append(pygame.transform.rotate(smokeImage, degrees))


	def update(self, data):
		if random.randint(0, self.intensity) == 0:
			Particle(data, random.choice(self.imgs), self.sourceCoords, self.particleLifespan, 
					 (random.randint(-10, 10), random.randint(-10, 10)), -1)
		if self.lifespan is not None and time.time() - self.birthTime > self.lifespan:
			self.kill()



class Explosion(pygame.sprite.Sprite):
	lifespan = 0.1
	def __init__(self, data, topleft, intensity):
		pygame.sprite.Sprite.__init__(self)
		self.add(data.particleSpawners)

		self.sourceCoords = topleft
		fireImage = data.loadImage('assets/particles/explosion.png')
		self.particleLifespan = 0.2
		self.birthTime = time.time()
		self.intensity = intensity

		SmokeSpawner(data, topleft, 1, 1.05)

		self.imgs = []
		for degrees in [0, 90, 180, 270]:
			self.imgs.append(pygame.transform.rotate(fireImage, degrees))

		sound.play('explosion%s' %(random.randint(1, 4)))


	def update(self, data):
		if time.time() - self.birthTime > Explosion.lifespan:
			self.kill()
		for i in range(self.intensity):
			Particle(data, random.choice(self.imgs), self.sourceCoords, self.particleLifespan, 
										 (random.randint(-10, 10), random.randint(-10, 10)), -1)



# class Shrapnel(pygame.sprite.Sprite):
# 	"""A hunk of metal that falls to the ground"""
# 	fallSpeed = 16
# 	acceleration = 20
# 	def __init__(self, data, topleft, image):
# 		pygame.sprite.Sprite.__init__(self)
# 		self.add(data.particles)

# 		self.image = image
# 		self.image.set_alpha(200)
# 		self.rect = self.image.get_rect(topleft = topleft)

# 		self.velocity = 0


# 	def update(self, data):
# 		self.velocity += Shrapnel.acceleration * data.dt
# 		if self.velocity > Shrapnel.fallSpeed:
# 			self.velocity = Shrapnel.fallSpeed

# 		self.rect.y += self.velocity * data.dt
		
# 		data.screen.blit(self.image, self.rect)