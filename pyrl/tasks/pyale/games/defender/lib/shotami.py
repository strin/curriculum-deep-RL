
from .enemi import enemi
from pygame import display
scr = display.get_surface()
scrrect = scr.get_rect()

def collidelist(bullet,lstenemi):
    for e,i in enumerate(lstenemi):
        if bullet.colliderect(i):
            sx,sy = i.x-bullet.x,i.y-bullet.y
            if bullet.msk.overlap(i.msk,(sx,sy)):
                return e
    return -1
    
class Shotami(list,object):
    
    def update(self):
        for f in self[:]:
            f.update()
            i = collidelist(f,enemi)
            if i>-1:
                enemi[i].shield -= f.pow
                f.pow = 0
                #~ self.remove(f)         
            if not scrrect.colliderect(f) or not f.pow:
                self.remove(f)
            else:
                f.render()
        #~ if self: return 1
        return 0


shotami     = Shotami() 
