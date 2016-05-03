    
from pygame import display
scr = display.get_surface()
scrrect = scr.get_rect()
from .ship import ship

    
def collideship(bullet):
    if bullet.colliderect(ship):
        sx,sy = ship.x-bullet.x,ship.y-bullet.y
        if bullet.msk.overlap(ship.msk,(sx,sy)):
            return True
    return False
        
class Shotenemi(list,object):
    
    def update(self):
        for f in self[:]:
            f.update()
            if collideship(f):
                ship.shield -= f.pow
                self.remove(f)
            elif not f.colliderect(scrrect) or not f.pow:
                self.remove(f)
            else:
                f.render()
        #~ if self: return 1
        return 0
                
shotenemi   = Shotenemi()
