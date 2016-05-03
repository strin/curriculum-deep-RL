
from pygame         import display, Rect, image, surfarray
from math import sin, cos, atan2, radians
scr = display.get_surface()
scrrect = scr.get_rect()
from .ship import ship

magnetradius = 0

class BonusBase(Rect,object):
    
    def __init__(self,img,x,y):
        self.img = img.copy()
        Rect.__init__(self,self.img.get_rect(center=(x,y)))
        self.ttl = 150
        self.deltaX = 0
        self.deltaY = 1
        self.X = self.x
        self.Y = self.y
        
    def update(self):
        sx,sy = ship.center
        deltax, deltay = -self.centerx+sx, -self.centery+sy
        if deltax**2+deltay**2 <= ship.magnetradius**2:
            tau = atan2(deltay,deltax)
            self.deltaX = cos(tau)*3
            self.deltaY = sin(tau)*3
        self.X += self.deltaX
        self.Y += self.deltaY
        self.x = int(self.X)
        self.y = int(self.Y)
            
        self.ttl -= 1
        alpha = 1.275*self.ttl
        Alpha = int(alpha)
        array = surfarray.pixels_alpha(self.img)
        array[array>alpha] = Alpha        
        if self.colliderect(ship):
            self.set()
            self.ttl = 0
    
    def render(self):
        scr.blit(self.img,self)

class BonusShield(BonusBase,object):
    
    @staticmethod
    def set():    
        ship.shield += 10

class BonusSetting(BonusBase,object):
    
    @staticmethod
    def set():    
        ship.settingbonus += 10
        
class BonusMagnet(BonusBase,object):
    
    @staticmethod
    def set():    
        ship.magnetradius += 50
        ship.magnetradius  = min((ship.magnetradius,200))
        
class Bonus(list,object):
    imgs = {'shield':image.load('img/shield0.png'),
            'setting':image.load('img/setting0.png'),
            'magnet':image.load('img/magnet0.png')}
    
    bonus = (lambda x,y : BonusShield(Bonus.imgs['shield'],x,y),
             lambda x,y : BonusSetting(Bonus.imgs['setting'],x,y),
             lambda x,y : BonusMagnet(Bonus.imgs['magnet'],x,y))
             
    shield  = staticmethod(bonus[0])
    setting = staticmethod(bonus[1])
    magnet  = staticmethod(bonus[2])
    
    def update(self):
        for f in self[:]:
            f.update()
            if f.ttl == 0: self.remove(f)
            if f.top > scrrect.bottom: self.remove(f)
            else: f.render()
        #~ if self: return 1
        return 0
    
    

bonus       = Bonus()
