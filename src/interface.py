import pygame as py 
import numpy as np
import pygame.surfarray as surfarray

class Display:
    def __init__(self):
        self.win = py.display.set_mode((250,200))
        py.display.set_caption("Gesture Recognition")
        py.font.init()
        self.font = py.font.SysFont("times",20)
         
    def input_image(self,features,image):
        py.display.blit(self.win) 

    def update(self,image,preds):
        self.win.fill((0,0,0))
        array = py.surfarray.pixels3d(self.win)
        image = image.transpose(0,1)
        image = np.rot90(image)
        
        array[60:158+30,30:158,1] = image
        array[60:158+30,30:158,0] = image
        array[60:158+30,30:158,2] = image
        del array
        if preds is not None:
            clas, labs = preds
            confidence = clas[0][np.argmax(clas[0])] 
            if confidence > 0.69:
                text = self.font.render(labs[ np.argmax(clas[0])]+": "+str(confidence)[:5],True,(200,200,200))
                trect = text.get_rect()
                trect.topleft = (30,160) 
                self.win.blit(text,trect)


        py.display.update()

        pass

if __name__ == '__main__':
    d = Display()
    pass