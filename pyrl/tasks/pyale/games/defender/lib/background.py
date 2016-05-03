
from pygame import display,image
from .layer import bulletlayer
scr = display.get_surface()
scrrect = scr.get_rect()
    
class Background(object):
    
    from .starsfield import Starsfield
    st           = Starsfield()
    update       = st.update
    earth        = image.load('img/earth.png')
    earthrect    = earth.get_rect(midbottom=scrrect.midbottom)
    
    @staticmethod
    def render():
        scr.fill(0)
        scr.blit(Background.st,(0,0))
        scr.blit(Background.earth,Background.earthrect)
        
