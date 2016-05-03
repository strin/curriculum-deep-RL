    
from math        import sin,cos,radians
from random      import random,randint,choice
from pygame      import display,Rect,image,mask
scr = display.get_surface()
from .shotenemi  import shotenemi
from .ship       import ship
from .particle   import explosion,Explosion
from .layer      import bulletlayer
from .bonus      import bonus
import os


class Red(Rect,object):
    
    
    class Bullet(Rect,object):
        
        img     = image.load('img/redbullet.png')
        rect    = img.get_rect()
        msk     = mask.from_surface(img,0)
        pow     = 10
        
        def __init__(self,midbottom):
            Rect.__init__(self,self.rect)
            self.midbottom  = midbottom
        
        def update(self):
            self.y += 10
        
        def render(self):
            bulletlayer.blit(self.img,self)
    
    img     = image.load('img/square.png')
    img2    = image.load('img/square2.png')
    msk     = mask.from_surface(img,0)
    
    def __init__(self,x,y):
        Rect.__init__(self,Red.img.get_rect(midbottom=(x,y)))
        self.axe        = x
        self.X, self.Y  = self.midbottom
        self.tau        = 0
        try:
            self.shield_ = 2. * float(os.environ['SHIELD_ENEMY'])
        except:
            self.shield_ = 30.
        self.foo        = 0
        self.tick       = 0
    
    def update(self):
        self.tick += 1
        self.tau += 1.8
        self.X = self.axe+sin(radians(self.tau))*80
        self.Y += 0.5
        self.midbottom = self.X, self.Y
        if self.left<ship.centerx<self.right and self.tick>=50:
            shotenemi.append(Red.Bullet(self.midbottom))
            self.tick = 0
    
    def render(self):
        if self.foo:
            self.foo -= 1
            scr.blit(Red.img2,self)
            return
        scr.blit(Red.img,self)

    @property
    def shield(self):
        return self.shield_

    @shield.setter
    def shield(self,n):
        self.shield_ = n
        self.foo = 5
        if n<0:
            explosion.update(explosion.Particles,self,(250,100,50),60,30)

class SMAlien(Rect,object):
    
    img = [image.load('img/space11.png'),image.load('img/space12.png'),image.load('img/space13.png'),image.load('img/space14.png'),image.load('img/space15.png')]
    msk = mask.from_surface(img[0],1)
    
    class Bullet(Rect,object):
        
        img     = image.load('img/oursin.png')
        rect    = img.get_rect()
        msk     = mask.from_surface(img,1)
        pow     = 5
        
        def __init__(self,midbottom):
            Rect.__init__(self,self.rect)
            self.midbottom  = midbottom
            self.dx         = random()-0.5
            self.X          = self.x
            self.dy         = 4+random()
            self.Y          = self.y
        
        def update(self):
            self.Y += self.dy
            self.X += self.dx
            self.x  = int(self.X)
            self.y  = int(self.Y)
        
        def render(self):
            bulletlayer.blit(self.img,self)
    
    
    def __init__(self,x,y):
        Rect.__init__(self,self.img[0].get_rect(midbottom=(x,y)))
        self.axe = x
        self.X, self.Y = self.midbottom
        self.tau = 0
        try:
            self.shield_ = float(os.environ['SHIELD_ENEMY'])
        except:
            self.shield_ = 30.
        self.foo = 0
        self.tick       = 0
        self.indeximg = 0
    
    def update(self):
        self.tick += 1
        self.tau += 1.8
        self.X = self.axe+sin(radians(self.tau))*80
        self.Y += 0.5
        self.midbottom = self.X, self.Y
        if self.left<ship.centerx<self.right and self.tick>=50:
            shotenemi.append(self.Bullet(self.midbottom))
            shotenemi.append(self.Bullet(self.midbottom))
            shotenemi.append(self.Bullet(self.midbottom))
            self.tick = 0
        self.foo = (self.foo+1)%10
        if not self.foo:
            self.indeximg = ((self.indeximg+1)%8)
    
    def render(self):
        scr.blit(self.img[abs(self.indeximg-4)],self)

    @property
    def shield(self):
        return self.shield_

    @shield.setter
    def shield(self,n):
        self.shield_ = n
        if n<0:
            explosion.update(explosion.Bulles,self,100,5)
            r = randint(0,10)
            if not r:
                bonus.append(bonus.setting(self.centerx,self.centery))
            elif 0<r<4:
                bonus.append(choice(bonus.bonus)(self.centerx,self.centery))

class SMAlien2(SMAlien,object):
    
    def update(self):
        self.tick += 1
        self.tau  += 1.8
        self.axe  += 0.5 if ship.centerx> self.axe else -0.5 if ship.centerx< self.axe else 0
        if self.axe < 100: self.axe = 100
        if self.axe >300 : self.axe = 300
        self.X = self.axe+sin(radians(self.tau))*80
        self.Y += 0.5
        self.midbottom = self.X, self.Y
        if self.left<ship.centerx<self.right and self.tick>=50:
            shotenemi.append(self.Bullet(self.midbottom))
            shotenemi.append(self.Bullet(self.midbottom))
            shotenemi.append(self.Bullet(self.midbottom))
            self.tick = 0
        self.foo = (self.foo+1)%10
        if not self.foo:
            self.indeximg = ((self.indeximg+1)%8)
    
