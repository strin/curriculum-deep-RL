    

from pygame import display,Surface,SRCALPHA,surfarray,Rect
from pygame import gfxdraw

scr = display.get_surface()
scrrect = scr.get_rect()

class Layer(Rect,object):
    
    def __init__(self,rect,alpha,delta):
        Rect.__init__(self,rect)
        self.surface = Surface(self.size,SRCALPHA)
        self.alpha   = alpha
        self.delta   = delta
    
    def render(self):
        scr.blit(self.surface,self)
        array = surfarray.pixels_alpha(self.surface)
        array[(self.alpha<array)&(array<self.alpha+self.delta)] = self.alpha
        array[array>self.alpha] = array[array>self.alpha]-self.delta
    
    def blit(self,*arg):
        self.surface.blit(*arg)
        
    def fill(self,*arg):
        self.surface.fill(*arg)
        
    def set_at(self,*arg):
        self.surface.set_at(*arg)
    

bulletlayer = Layer(scrrect,0,50)
osdlayer  = Layer((0,500,400,80),100,10)
