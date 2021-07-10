import torch
import torch.nn as nn

#def vgg_block_single(in_ch, out_ch, kernel_size=3, padding=1):
#    return nn.Sequential(
#        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
#        nn.ReLU(),
#        nn.MaxPool2d(kernel_size=2, stride=2)
#        )
#
#def vgg_block_double(in_ch, out_ch, kernel_size=3, padding=1):
#    return nn.Sequential(
#        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
#        nn.ReLU(),
#        nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding),
#        nn.ReLU(),
#        nn.MaxPool2d(kernel_size=2, stride=2)
#        )
#
#
#class VGG11(nn.Module):
#    def __init__(self, in_ch, num_classes):
#        super().__init__()
#
#        self.conv_block1 =vgg_block_single(in_ch,64)
#        self.conv_block2 =vgg_block_single(64,128)
#
#        self.conv_block3 =vgg_block_double(128,256)
#        self.conv_block4 =vgg_block_double(256,512)
#        self.conv_block5 =vgg_block_double(512,512)
#
#        self.fc_layers = nn.Sequential(
#            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True), nn.Dropout(),
#            nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(),
#            nn.Linear(4096, num_classes)
#        )
#
#    def forward(self, x):
#
#        x=self.conv_block1(x)
#        x=self.conv_block2(x)
#
#        x=self.conv_block3(x)
#        x=self.conv_block4(x)
#        x=self.conv_block5(x)
#
#        x=x.view(x.size(0), -1)
#
#        x=self.fc_layers(x)
#
#        return x


def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        nn.BatchNorm2d(chann_out),
        nn.ReLU()
    )
    return layer


def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):
    layers = [conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list))]
    layers += [nn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]
    return nn.Sequential(*layers)


def vgg_fc_layer(size_in, size_out):
    layer = nn.Sequential(
        nn.Linear(size_in, size_out),
        nn.BatchNorm1d(size_out),
        nn.ReLU()
    )
    return layer


class VGG16(nn.Module):
    def __init__(self, n_classes=1000):
        super(VGG16, self).__init__()

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([3, 64], [64, 64], [3, 3], [1, 1], 2, 2)
        self.layer2 = vgg_conv_block([64, 128], [128, 128], [3, 3], [1, 1], 2, 2)
        self.layer3 = vgg_conv_block([128, 256, 256], [256, 256, 256], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer4 = vgg_conv_block([256, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer5 = vgg_conv_block([512, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)

        # FC layers
        self.layer6 = vgg_fc_layer(7 * 7 * 512, 4096)
        self.layer7 = vgg_fc_layer(4096, 4096)

        # Final layer
        self.layer8 = nn.Linear(4096, n_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        vgg16_features = self.layer5(out)
        out = vgg16_features.view(out.size(0), -1)
        out_fc1 = self.layer6(out)
        out_fc2 = self.layer7(out_fc1)
        out = self.layer8(out_fc2)

        return out_fc2, out


