class Viewer:
    from PIL import Image
    from examples.Event_camera_sim import Event_simulator
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    import time
    event_sim=Event_simulator()
    def __init__(self,height=64,width=64):
        # self.observation=observeation
        # self.dimensions = (100, 100)
        self.height=height
        self.width=width
        self.RGB=self.np.zeros((self.height, self.width, 3),dtype=self.np.uint8)
        self.event_figure=self.plt.figure(num='event_viewer')

    def RGB_view(self,observeation):

        resized = self.Image.fromarray(observeation['0']['rgb'].astype('uint8'), 'RGBA')
        self.plt.imshow(resized)
        self.plt.show(block=False)
        self.plt.pause(0.01)



    def Event_view(self,pos, neg,event_frame):
        # self.plt.figure(num="event viewer")
        img = self.np.zeros((self.height,self.width))
        # img = img + pos * 0.5 - neg * 0.5
        # img =  pos * 0.5 - neg * 0.5
        img=event_frame
        # img[pos]=1
        # img[neg]=-1
        # img=img+pos*255+neg*(-255)
        img=self.cv2.resize(img, (400, 400))
        self.cv2.imshow('ev frame', img)
        self.time.sleep(0.01)

        # self.plt.imshow(img)
        # # self.plt.show(block=False)
        # self.plt.pause(0.1)

    def RGB_get(self,observeation):
        self.RGB=self.Image.fromarray(observeation['0']['rgb'].astype('uint8'), 'RGBA')
        return  self.RGB
