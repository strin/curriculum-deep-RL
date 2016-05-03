    # -*- coding: utf-8 -*-
    
from pygame         import image,Rect,mask,key,font,draw,Color,transform,surfarray
from pygame         import gfxdraw
from pygame.locals  import *
from pygame         import display
scr = display.get_surface()
scrrect = scr.get_rect()
#font.init()
#police = font.Font('Roboto.ttf',10)
#police.set_bold(1)
#bonuspolice = font.Font('Roboto.ttf',10)
from .particle     import explosion
from .layer        import osdlayer
from .shotami      import shotami
from .layer        import bulletlayer
from .enemi        import enemi
import os



class Bonus(object):
    
    def __init__(self,img,x,y,value):
        self.img   = img
        self.rect  = img.get_rect(topleft=(x,y))
        self.update(value)
    
    def update(self,value):
        self.value  = value
        osdlayer.blit(self.img,self.rect)
        draw.rect(osdlayer.surface,(200,200,200),self.rect,1)
        #self.valuefx = bonuspolice.render(str(value),1,(255,255,255),(0,0,0))
        #self.valuefxrect = self.valuefx.get_rect(bottomright=self.rect.bottomright).move(osdlayer.topleft)
    
    def render(self):
        #scr.blit(self.valuefx,self.valuefxrect)
        pass


class DoubleLazer(object):
        
    cadence   = 10
    tempmax   = 2000.
    deltatemp = 50
    deltaref  = 1
    t         = cadence
    temper    = 0
    
    class Bullet(Rect,object):
        img     = image.load('img/lazer1.png')
        msk     = mask.from_surface(img,1)
        rect    = img.get_rect()
        pow     = 10
        
        def __init__(self,pos):
            Rect.__init__(self,self.rect)
            self.midbottom = pos
        
        def update(self):
            self.y  -= 10
            self.pos = self.x, self.y
        
        def render(self):
            bulletlayer.blit(self.img,self)
    
    def __init__(self,get_pos1,get_pos2):
        self.get_pos1 = get_pos1
        self.get_pos2 = get_pos2
        
    def shot(self):
        if DoubleLazer.t >= DoubleLazer.cadence and DoubleLazer.temper<DoubleLazer.tempmax:
            DoubleLazer.t = 0
            shotami.append(DoubleLazer.Bullet(self.get_pos1()))
            shotami.append(DoubleLazer.Bullet(self.get_pos2()))
            DoubleLazer.temper += DoubleLazer.deltatemp
    
    def update(self):
        DoubleLazer.t += 1
        if  DoubleLazer.temper >= DoubleLazer.deltaref:
            DoubleLazer.temper -= DoubleLazer.deltaref
        else:
            DoubleLazer.temper  = 0
            
    def clear(self):
        DoubleLazer.t       = DoubleLazer.cadence
        DoubleLazer.temper  = 0

class Torpedo0(Rect,object):
    
    
    class Blast(Rect,object):
        img = image.load('img/t0blast.png')
        msk = mask.Mask((0,0))
        
        def __init__(self,pos):
            Rect.__init__(self,pos,(0,0))
            self.pos       = pos
            self.radius    = 0
            self.radiusmax = 500
            self.pow       = 40
            self.step      = self.radiusmax/self.pow
            self.img       = None
        
        def update(self):
            delta        = self.pow/80.
            self.radius += self.step
            self.pow    -= 1
            r            = (int(self.radius),)*2
            self.img     = transform.smoothscale(Torpedo0.Blast.img,r)
            array = surfarray.pixels_alpha(self.img)
            array[:] = array[:]*delta
            self.size    = r
            self.center  = self.pos
            for f in enemi:
                x,y = f.center
                h2 = (self.centerx-x)**2+(self.centery-y)**2
                if h2 < (self.radius/2.)**2:
                    f.shield -= 1
                
        
        def render(self):
            bulletlayer.blit(self.img,self)
            
        
    img     = image.load('img/torpedo1.png')
    msk     = mask.from_surface(img,1)
    rect    = img.get_rect()
    pow_    = 50
    ttl     = 100
    
    def __init__(self,pos):
        Rect.__init__(self,self.rect)
        self.midbottom = pos
    
    def update(self):
        self.y   -= 3
        self.pos  = self.x, self.y
        self.ttl -= 1
        if self.ttl == 0:
            self.pow = 0
    
    def render(self):
        bulletlayer.blit(self.img,self)
    
    def blast(self):
        shotami.append(Torpedo0.Blast(self.center))
    
    @property
    def pow(self):
        return self.pow_
    @pow.setter
    def pow(self,value):
        if value == 0:
            self.blast()
            self.pow_ = value
        

torpedo1 = lambda pos: Torpedo0(pos)
class LanceTorpille(object):
        
    cadence   = 50
    #~ tempmax   = 2000
    #~ deltatemp = 50
    #~ deltaref  = 1
    t         = cadence
    #~ temper    = 0

    
    def __init__(self,get_pos):
        self.get_pos = get_pos
        
    def update(self):
        LanceTorpille.t += 1
        
    def shot(self,torpedo):
        if LanceTorpille.t >= LanceTorpille.cadence:
            LanceTorpille.t = 0
            shotami.append(torpedo(self.get_pos()))
            return 1
            
    def clear(self):
        LanceTorpille.t       = LanceTorpille.cadence

class Ship(Rect,object):
    
    img          = image.load('img/xwing.png')
    rect         = img.get_rect()
    img2         = image.load('img/xwing2.png')
    msk          = mask.from_surface(img,1)
    lazertempfx  = image.load('img/lazertemp.png')
    osdlayer.blit(lazertempfx,(10,20))
    osdlayer.fill((100,100,200,200),(10,0,380,18))
    #SHIELDTEXT   = police.render('SHIELD',1,(250,250,250))
    #SHIELDRECT   = SHIELDTEXT.get_rect(midleft=(15,509))
    #LAZERTEXT    = police.render('LAZER°',1,(250,250,250))
    #LAZERRECT    = LAZERTEXT.get_rect(midleft=(15,529))
    

    def __init__(self):
        self.memgun1accum    = 3
        try:
            self.memshield_ = int(os.environ.get('SHIELD_SHIP'))
        except:
            self.memshield_ = 5000

        self.memshieldmax = self.memshield_
        #~ les bonus doivent avoir un offest en fonction de osdlayer
        self.settingbonus_   = Bonus(image.load('img/setting0.png'),10,42,0)
        self.memsettingbonus = self.settingbonus
        self.loader1_   = Bonus(image.load('img/torpedo1.png'),50,42,8)
        self.memloader1 = self.loader1
        Rect.__init__(self,self.rect)
        self.lazer = DoubleLazer(lambda: self.midleft,lambda: self.midright)
        self.lancetorpille = LanceTorpille(lambda: self.midtop)
        self.clear()
            
    
    def update(self,ev):
        
        def dir_update(ev):                
            if ev.type == KEYDOWN:
                self.dirx += (ev.key == K_RIGHT) - (ev.key == K_LEFT)
                self.diry +=  (ev.key == K_DOWN) - (ev.key == K_UP)
                if ev.key == K_SPACE:
                    self.shotbutton = True
                if ev.key == K_F1 and self.loader1:
                    if self.lancetorpille.shot(torpedo1):
                        self.loader1 -= 1
            elif ev.type == KEYUP:
                self.dirx += (ev.key == K_LEFT) - (ev.key == K_RIGHT)
                self.diry +=   (ev.key == K_UP) - (ev.key == K_DOWN)
                if ev.key == K_SPACE:
                    self.shotbutton = False
            elif ev.type == JOYAXISMOTION:
                value = int(ev.value)
                if value:
                    if ev.axis == 0: self.dirx = value
                    elif ev.axis == 1: self.diry = value
                else:
                    if ev.axis == 0: self.dirx = value
                    elif ev.axis == 1: self.diry = value
            elif ev.type == JOYBUTTONDOWN:
                if ev.button == 7:
                    self.shotbutton = True
            elif ev.type == JOYBUTTONUP:
                if ev.button == 7:
                    self.shotbutton = False
                
        def gun_update():
            self.lazer.update()
            if self.shotbutton:
                self.lazer.shot()
            
        
        if self.shield < 0:
            self.topleft = -1,-1
            self.size = 0,0
            return 0
        else:
            dir_update(ev)
            gun_update()
            self.lancetorpille.update()
            self.vitx    = round(sorted((-self.vmax,self.vitx+self.acc*self.dirx,self.vmax))[1],2)
            self.vity    = round(sorted((-self.vmax,self.vity+self.acc*self.diry,self.vmax))[1],2)
            self.X      += self.vitx
            self.Y      += self.vity
            self.X       = sorted((scrrect.left,self.X,scrrect.right-self.w-1))[1]
            self.Y       = sorted((scrrect.top,self.Y,scrrect.bottom-self.h))[1]
            self.topleft = self.X,self.Y
            self.render()
            self.magnetradius -= bool(self.magnetradius)*0.25
            return 1
        
    def render(self):
        #~ draw magnetradius
        if self.magnetradius:
            #~ gfxdraw.aacircle(scr,self.centerx,self.centery,int(self.magnetradius),(150,130,110,int(self.magnetradius)))
            m = image.load('img/magnetichalo.png')
            m = transform.smoothscale(m,[int(self.magnetradius*2)]*2)
            scr.blit(m,m.get_rect(center=self.center))
        #~ draw shieldbar
        r = draw.rect(scr,(50,50,50),(10,500,380,18),1)
        osdlayer.fill((100,100,200,self.shieldfxttl*4+150),(10,0,self.shield_/self.shieldmax*380,18))
        #scr.blit(self.SHIELDTEXT,self.SHIELDRECT)
        #~ draw lazerstate
        r = draw.rect(scr,(50,50,50),(10,520,380,18),1)
        osdlayer.blit(self.lazertempfx,(10,20),(0,0,DoubleLazer.temper/DoubleLazer.tempmax*380,18))
        #scr.blit(self.LAZERTEXT,self.LAZERRECT)
        #~ draw bonusbar
        self.settingbonus_.render()
        self.loader1_.render()
        #~ draw shieldcircle
        if self.shieldfxttl:
            self.shieldcolor.a = int(4*self.shieldfxttl)
            gfxdraw.filled_circle(scr, self.centerx, self.centery, self.w, self.shieldcolor)
            self.shieldfxttl -= 1
        if self.foo:
            self.foo -= 1
            scr.blit(self.img2,ship)
            return
        #~ draw ship
        scr.blit(self.img,ship)
    
    @property
    def shield(self):
        return self.shield_
    
    @shield.setter
    def shield(self,value):
        if self.shield_ < value: self.shieldcolor = Color(100,100,200,0)
        else: self.shieldcolor = Color(200,100,100,0)
        self.shield_    = value if value <= self.shieldmax else self.shieldmax
        if value < 0:
            explosion.update(explosion.Particles,self,(250,250,250),100)
        self.foo         = 5
        self.shieldfxttl = 25
        
    
    def clear(self):
        self.size           = self.rect.size
        self.midbottom      = (200,490)
        self.acc            = 0.5
        self.vmax           = 3
        self.vitx           = 0
        self.vity           = 0
        k = key.get_pressed()
        self.dirx           = k[K_RIGHT]-k[K_LEFT]
        self.diry           = k[K_DOWN]-k[K_UP]
        self.Lazeraccum     = self.memgun1accum
        self.shotbutton     = False
        self.X              = self.x
        self.Y              = self.y
        self.shieldmax      = self.memshieldmax
        self.shield_        = self.memshield_
        self.foo            = 0
        self.shieldfxttl    = 0
        self.settingbonus   = self.memsettingbonus
        self.loader1   = self.memloader1
        self.magnetradius   = 0
        self.lazer.clear()
    
    @property
    def settingbonus(self):
        return self.settingbonus_.value
    @settingbonus.setter
    def settingbonus(self,value):
        self.settingbonus_.update(value)
    @property
    def loader1(self):
        return self.loader1_.value
    @loader1.setter
    def loader1(self,value):
        self.loader1_.update(value)
        
ship = Ship()
        


