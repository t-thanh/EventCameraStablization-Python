class Helper():
    # def __init__(self):    #
    import torch


    def Get_angular_velocities(self,observation):
        self.observation = observation
        self.state = observation[str(0)]["state"]
        cur_ang_vel = self.state[13:16]
        return cur_ang_vel

    def SavemyModel(self,model,path='mymodel.pkl'):
        print('model saved succesfully')
        self.torch.save(model, path)

    def LoadMyModel(self,path='mymodel.pkl'):
        model = self.torch.load(path)
        return model
