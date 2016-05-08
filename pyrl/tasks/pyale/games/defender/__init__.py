from pygame         import *
scr         = display.set_mode((400,600), 0, 32)
scrrect     = scr.get_rect()
#font.init()
#police = font.Font('Roboto.ttf',50)
from lib.enemi      import enemi
from lib.shotenemi  import shotenemi
from lib.shotami    import shotami
from lib.ship       import ship
from lib.particle   import explosion
from lib.layer      import bulletlayer, osdlayer
from lib.background import Background
from lib.levelmess  import Levelmess
from lib.bonus      import bonus

enemi.shot = 0

joystick.init()
if joystick.get_count(): joystick.Joystick(0).init()
    
class Game(object):
    
    levels     = ['Level1']
    levelcount = 1
    ck0        = time.Clock()
    exploded = False
    
    
    def run(self):
        while self.levels:
            levelname = self.levels[0]
            from lib.levels import Level1
            level = Level1()
            #level     = __import__('lib.levels',None,None,[levelname])
            #level     = getattr(level,levelname)
            Game().clear(level)
            display.set_caption('Defender <Level {0}>'.format(self.levelcount))
            Levelmess.update(level.__doc__)
            endlevelflag = 0
            while not endlevelflag:
                for ev in event.get():
                    if ev.type == QUIT: exit()
                    statuslevel = level.update()
                    Background.render()
                    bulletlayer.render()
                    osdlayer.render()
                    explosion.render()
                    statusship      = ship.update(ev)
                    Background.update()
                    statusmess = Levelmess.render()
                    if not any((statuslevel,explosion.status)):
                        self.levels.pop(0)
                        self.levelcount += 1
                        endlevelflag = True
                        display.update()
                        display.flip() # important for saving "exploded" information
                        break
                    shotami.update()
                    enemi.update()
                    bonus.update()
                    shotenemi.update()
                    if not any((statusship,explosion.status)):
                        Levelmess.update('Level Reloaded')
                        self.clear(level)
                        self.exploded = True
                        display.update()
                        display.flip() # important for saving "exploded" information
                        return
                display.flip()
            scr.fill(0)
            Levelmess.update('Thanks for testing')
            Levelmess.render()
            display.update()
    
    def clear(self,level):
        enemi[:] = []
        shotami[:] = []
        shotenemi[:] = []
        ship.clear()
        level.clear()
    
        

game = Game()
game.run()
    
