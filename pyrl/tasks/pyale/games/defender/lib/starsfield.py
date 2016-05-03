from pygame import Surface,display,PixelArray,time,SRCALPHA
from random import randrange,choice

class Starsfield(Surface,object):
    """"""
    def __init__(self):
        scrsize = self.w,self.h = display.get_surface().get_size()
        Surface.__init__(self,scrsize,SRCALPHA)
        self.foo = 0
        self.l0 = []
        self.l1 = []
        self.l2 = []
    
    def update(self):
        self.foo += 1      
        pixar = PixelArray(self)
        x = randrange(100)
        
        if not self.foo&3:
            self.l0.append([[randrange(self.w),0],(randrange(0x33),)*3])
            for s in self.l0[:]:
                pixar[s[0][0]][s[0][1]] = 0
                s[0][1] += 1
                if s[0][1]<self.h: pixar[s[0][0]][s[0][1]] = s[1] if randrange(500) else 0x9999ff
                else: self.l0.pop(0)
        
        if self.foo&1:
            if x<10: self.l1.append([[randrange(self.w),0],(randrange(0x33,0x99),)*3])
            for s in self.l1[:]:
                pixar[s[0][0]][s[0][1]] = 0
                s[0][1] += 1
                if s[0][1]<self.h: pixar[s[0][0]][s[0][1]] = s[1]
                else: self.l1.pop(0)
        
        if not x: self.l2.append([[randrange(self.w),0],(randrange(0x99,0xff),)*3])
        for s in self.l2[:]:
            pixar[s[0][0]][s[0][1]] = 0
            s[0][1] += 1
            if s[0][1]<self.h: pixar[s[0][0]][s[0][1]] = s[1]
            else: self.l2.pop(0)

if __name__ == "__main__":
    scr = display.set_mode((400,500))
    st = Starsfield()
    while True:
        scr.fill(0)
        st.update()
        scr.blit(st,(0,0))
        display.flip()
        time.wait(1)
    
