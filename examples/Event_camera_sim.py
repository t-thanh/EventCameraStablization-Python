class Event_simulator:
    from PIL import Image
    import matplotlib.pyplot as plt
    import cv2
    import torch
    import numpy as np
    def __init__(self,frame_dimensions=[64,64]):
        # self.observation=observeation
        self.height=frame_dimensions[0]
        self.width=frame_dimensions[1]
        self.Current_frame= self.np.zeros((self.height, self.width, 3),
                     dtype=self.np.uint8)
        self.Previous_frame=self.Current_frame
        self.spike_tensor=self.torch.zeros((2,  self.height,  self.width, 1))
        # self.pos_event=self.torch.zeros((self.height,self.width))
        # self.neg_event = self.torch.zeros((self.height, self.width))
        self.event_frame=self.np.zeros((self.height, self.width),dtype=self.np.uint8)
    def Event_From_Adjecent_Frames(self,Current_frame,Previous_frame):

        self.Current_frame=Current_frame
        self.Previous_frame=Previous_frame
        current_frame_gray = self.cv2.cvtColor(self.np.array(self.Current_frame), self.cv2.COLOR_RGB2GRAY)
        previous_frame_gray = self.cv2.cvtColor(self.np.array(self.Previous_frame),self.cv2.COLOR_RGB2GRAY)
        time_diffrent_frame = self.np.double(current_frame_gray) - self.np.double(previous_frame_gray)
        time_diffrent_frame = self.torch.tensor(time_diffrent_frame)
        polarity = self.np.sign(time_diffrent_frame)
        pos = polarity[time_diffrent_frame > 0.3]
        neg = polarity[time_diffrent_frame < -0.3]
        ev_pol = self.np.hstack((pos, neg * 0))
        xy_pos = self.np.where(time_diffrent_frame > 0.3)
        xy_neg = self.np.where(time_diffrent_frame < -0.3)
        ev_xy = self.np.hstack((self.np.asarray(xy_pos), self.np.asarray(xy_neg)))
        self.spike_tensor[self.np.int8(ev_pol), ev_xy[0, :], ev_xy[1, :]] = 1
        self.pos_event=self.np.array(self.np.squeeze(self.spike_tensor[0]))
        self.neg_event=self.np.array(self.np.squeeze(self.spike_tensor[1]))
        time_diffrent_frame_temp=self.np.array(time_diffrent_frame)
        time_diffrent_frame[time_diffrent_frame > 0.3] = 1
        time_diffrent_frame[time_diffrent_frame < -0.3] = -1
        time_diffrent_frame[abs(time_diffrent_frame)!=1]=0
        # time_diffrent_frame[(time_diffrent_frame<0.3 and time_diffrent_frame>-0.3)]=0
        event_frame_temp = self.np.zeros((self.height, self.width))+self.np.array(time_diffrent_frame)+0.5 #image of 1, and 0
        event_frame=self.np.multiply(event_frame_temp,time_diffrent_frame_temp)
        # Take only region of logo from logo image.
        # event_frame = self.cv2.bitwise_and(time_diffrent_frame_temp,event_frame_temp)

        self.event_frame=event_frame_temp
        return time_diffrent_frame
        # self.pos_event=xy_pos
        # self.neg_event=xy_neg
        # self.pos_value=pos
        # self.neg_value=neg
        # try:
        #     time_diffrent_frame=time_diffrent_frame[time_diffrent_frame > 0.3].reshape((200,200)) +time_diffrent_frame[time_diffrent_frame < -0.3].reshape((200,200))
        # except:
        #     time_diffrent_frame = time_diffrent_frame[time_diffrent_frame > 0.3].reshape((200,200))
