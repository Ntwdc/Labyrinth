import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, n_input_layers):
        super(CNN,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(n_input_layers, 64,kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(64))
    
        self.layer2 = nn.Sequential( 
            nn.Conv2d(64, 64,kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(64))
        
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, padding=0))
            
        self.layer4 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128))

        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 128,kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128))
            
        self.layer6 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, padding=0))
            
        self.layer7 = nn.Sequential(
            nn.Conv2d(128, 256,kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(256))
            
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 256,kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(256))
            
        self.layer9 = nn.Sequential(
            nn.Conv2d(256, 256,kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(256))
            
        self.layer10 = nn.UpsamplingBilinear2d(scale_factor=2)
            
        self.layer11 = nn.Sequential(
            nn.Conv2d(256, 128,kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128))
            
        self.layer12 = nn.Sequential(
            nn.Conv2d(128, 128,kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128))
            
        self.layer13 = nn.Sequential(
            nn.Conv2d(128, 128,kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128))
            
        self.layer14 = nn.UpsamplingBilinear2d(scale_factor=2)
            
        self.layer15 = nn.Sequential(
            nn.Conv2d(128, 64,kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(64))
            
        self.layer16 = nn.Sequential(
            nn.Conv2d(64, 64,kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(64))
            
        self.layer17 = nn.Sequential(
            nn.Conv2d(64, 1,kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(1))
        
        self.layer18 = nn.Softmax(dim=1)
            
            
    def forward(self,x):
        x = self.layer1(x)  
        x = self.layer2(x)  
        x = self.layer3(x)  
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)  
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)  
        x = self.layer10(x)  
        x = self.layer11(x)  
        x = self.layer12(x)  
        x = self.layer13(x)  
        x = self.layer14(x)  
        x = self.layer15(x)  
        x = self.layer16(x)
        x = self.layer17(x) 
        x = x.view(1, -1)
        x = self.layer18(x)
        x = x.reshape(1, 180, 240)
        
        return x