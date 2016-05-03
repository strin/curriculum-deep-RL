    
    
from pygame import display,image,surfarray
scr = display.get_surface()
scrrect = scr.get_rect()
from .layer import bulletlayer

from math   import sin,cos,radians
from random import random,randint,choice


class Explosion(list,object):
    
    class Particles(list,object):    
        
        class Particle(object):
            
            def __init__(self,x0,y0,x1,y1,c0,c1,s):
                self.x               = x0
                self.y               = y0
                self.dx              = (x1-x0)/s
                self.dy              = (y1-y0)/s
                self.r,self.g,self.b = c0
                r1,g1,b1             = c1
                self.dr              = (r1-self.r)/s
                self.dg              = (g1-self.g)/s
                self.db              = (b1-self.b)/s
                self.s               = s
                
            
            def render(self):
                bulletlayer.set_at((int(self.x),int(self.y)),(int(self.r),int(self.g),int(self.b)))
                self.x += self.dx
                self.y += self.dy+1
                self.r += self.dr
                self.g += self.dg
                self.b += self.db
                self.s -= 1
    
        def __init__(self,obj,couleur,distance,quantite=None):
            x,y = obj.center
            if not quantite: quantite = obj.msk.count()
            for i in range(quantite):
                i  = radians(random()*360)
                x0 = cos(i)*randint(0,obj.w//4)+x
                y0 = sin(i)*randint(0,obj.h//4)+y
                x1 = cos(i)*randint(0,obj.inflate(distance,distance).h)+x
                y1 = sin(i)*randint(0,obj.inflate(distance,distance).w)+y
                ps = randint(10,100)
                self.append(Explosion.Particles.Particle(x0,y0,x1,y1,couleur,(200,200,200),ps))
        
        
        
    class Bulles(list,object):
        
        class Bulle(object):    
        
            imgs = [image.load('img/Bubble.png')]
            
            def __init__(self,x0,y0,x1,y1,s):
                self.x               = x0
                self.y               = y0
                self.dx              = (x1-x0)/s
                self.dy              = (y1-y0)/s 
                self.s               = s
                self.img             = choice(Explosion.Bulles.Bulle.imgs).copy()
                
            
            def render(self):
                bulletlayer.blit(self.img,(int(self.x),int(self.y)))
                self.x += self.dx
                self.y += self.dy-1
                self.s -= 1
                array = surfarray.pixels_alpha(self.img)
                array[array>0] = array[array>0]-1
        
        def __init__(self,obj,distance,quantite=None):
            x,y = obj.center
            if not quantite: quantite = obj.msk.count()
            for i in range(quantite):
                i  = radians(random()*360)
                x0 = cos(i)*randint(0,obj.w)+x
                y0 = sin(i)*randint(0,obj.h)+y
                x1 = cos(i)*randint(obj.w,obj.inflate(distance,distance).h)+x
                y1 = sin(i)*randint(obj.h,obj.inflate(distance,distance).w)+y
                ps = randint(100,255)
                self.append(Explosion.Bulles.Bulle(x0,y0,x1,y1,ps))
        
    def update(self,type,*args):
        self.extend(type(*args))
            
    def render(self):
        for i in self[:]:
            i.render()
            if i.s == 0:
                self.remove(i)
    
    @property
    def status(self):
        return bool(self)
                
explosion = Explosion()
