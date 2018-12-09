import torch.nn as nn
import torch.utils.model_zoo as model_zoo


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}
settings =   {  'input_space': 'RGB',
            'input_size': [3, 450, 450],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
             }

class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )            
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(    
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(    
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.classifier = nn.Sequential(
#             nn.Dropout(),
            nn.ReLU(inplace=True),
#             nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, model_root=None, **kwargs):
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet'], model_root))
    return model

class TCNN_Alex(AlexNet):
    def __init__(self,num_classes):
        super(TCNN_Alex,self).__init__()
        self.num_classes = num_classes
    
    def energy_layer(self,x):
        x_size = x.size()
        avgpool = nn.AvgPool2d(kernel_size = x_size[2],stride = 1).cuda()
        out = avgpool(x)
        return out
    def fc1_layer(self,x):
        x_size = x.size()
        x = x.view(x.size(0), x_size[1])
        fc = nn.Linear(x_size[1],4096).cuda()
        #print(x.type())
        out = fc(x)
        return out
    def forward(self,x):
        x = self.conv1(x)
        #print(x.size())
        x = self.conv2(x)
        #print(x.size())
        x = self.conv3(x)
        #print(x.size())
        e = self.energy_layer(x)
        #print(e.size())
        
        x = self.fc1_layer(e)
        #print(x.size())
        x = self.classifier(x)
        #print(x.size())
#         torch.Size([2, 64, 55, 55])
#         torch.Size([2, 192, 27, 27])
#         torch.Size([2, 384, 27, 27])
#         torch.Size([2, 384, 1, 1])
#         torch.cuda.FloatTensor
#         torch.Size([2, 4096])
#         torch.Size([2, 2])
        return x
        

def tcnn(num_classes=1000,pretrained=False, model_root=None,**kwargs):
    model = TCNN_Alex(num_classes,**kwargs)
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']
    if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['alexnet'], model_root))
        #调用模型 
        model_dict = model_zoo.load_url(model_urls['alexnet'], model_root)
        pretrained_dict = model.state_dict()
        k1 = ['features.0.weight', 'features.0.bias', 'features.3.weight', 'features.3.bias', 'features.6.weight', 'features.6.bias', 'features.8.weight', 'features.8.bias', 'features.10.weight', 'features.10.bias', 'classifier.1.weight', 'classifier.1.bias', 'classifier.4.weight', 'classifier.4.bias', 'classifier.6.weight', 'classifier.6.bias']
        k2 = ['conv1.0.weight', 'conv1.0.bias', 'conv2.0.weight', 'conv2.0.bias', 'conv3.0.weight', 'conv3.0.bias', 'conv4.0.weight', 'conv4.0.bias', 'conv5.0.weight', 'conv5.0.bias', 'fc1.weight', 'fc1.bias', 'classifier.1.weight', 'classifier.1.bias', 'classifier.3.weight', 'classifier.3.bias']
        for i in range(len(k1)-2):
            pretrained_dict[k2[i]] = model_dict[k1[i]]

        # 2. overwrite entries in the existing state dict
#         model_dict.update(pretrained_dict)
#         # 1. filter out unnecessary keys
#         pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#         # 2. overwrite entries in the existing state dict
#         model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(pretrained_dict)
    return model    