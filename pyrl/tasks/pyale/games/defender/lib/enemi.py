
from pygame import display
scr = display.get_surface()
scrrect = scr.get_rect()

class Enemi(list,object):
    def __init__(self):
        self.shot = 0
    
    def update(self):
        for f in self[:]:
            f.update()
            if f.shield < 0:
                self.shot += 1
                self.remove(f)
            if f.top > scrrect.bottom: self.remove(f)
            else: f.render()
        #~ if self: return 1
        return 0

enemi       = Enemi()
