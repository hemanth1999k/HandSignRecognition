import pygame as py 
import numpy as np
import pygame.surfarray as surfarray

class Display:
    def __init__(self):
        self.win = py.display.set_mode((250,200))
        py.display.set_caption("Gesture Recognition")
        py.font.init()
        self.font = py.font.SysFont("times",20)
        self.selecting_model = True
        self.model_type = 0
        self.availables = {
            0:[5,8,10],
            1:[5,8,10],
            2:[5,8,10],
        }
         
    def input_image(self,features,image):
        py.display.blit(self.win) 

    def update(self,image,preds):
        self.win.fill((0,0,0))
        return_command = None
        for event in py.event.get():
            if event.type == py.QUIT:
                exit()
            if event.type == py.KEYDOWN:
                if event.key == py.K_d:
                    self.selecting_model = True
                    self.model_type = 0
                    pass                    
                if event.key == py.K_f:
                    self.selecting_model = True
                    self.model_type = 1
                    pass
                if event.key == py.K_c:
                    self.selecting_model = True
                    self.model_type = 2
                    pass
                if self.selecting_model:
                    if event.key == py.K_1:
                        return_command = [self.model_type,self.availables[self.model_type][0]]
                        self.selecting_model = False
                        pass
                    if event.key == py.K_2:
                        return_command = [self.model_type,self.availables[self.model_type][1]] 
                        self.selecting_model = False
                        pass
                    if event.key == py.K_3:
                        return_command = [self.model_type,self.availables[self.model_type][2]] 
                        self.selecting_model = False
                        pass
                    

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
        return return_command

if __name__ == '__main__':
    d = Display()
    pass