
from pygame import display,font
font.init()
#police = font.Font('Roboto.ttf',50)
scr = display.get_surface()
scrrect = scr.get_rect()

class Levelmess(object):
    
    t = 0
    
    @staticmethod
    def update(mess):
        #Levelmess.mess     = police.render(mess,1,(200,200,200))
        #Levelmess.messrect = Levelmess.mess.get_rect(center=scrrect.center)
        #Levelmess.t        = 0
        pass
        
    
    @staticmethod
    def render():
        return 1
        #Levelmess.t += 1
        #if Levelmess.t == 125:
        #    Levelmess.update('')
        #    Levelmess.t = 0
        #    return 0
        #scr.blit(Levelmess.mess,Levelmess.messrect)
        #return 1

Levelmess.update('')
