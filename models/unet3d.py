import torch
import torch.nn as nn
import pytorch_lightning as pl


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.conv3 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm3 = nn.InstanceNorm3d(out_channels)

    def forward(self, x):
        x2 = self.relu(self.norm(self.conv(x)))
        x3 = self.relu(self.norm2(self.conv2(x2)))
        x4 = self.relu(self.norm3(self.conv3(x3)))
        x_out = x2+x4
        return x_out 


# Encoder block (Conv block + downsampling)
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.downsample = nn.MaxPool3d(2, stride=2) 

    def forward(self, x):
        x = self.conv_block(x)
        x_downsampled = self.downsample(x)
        return x_downsampled, x


class Encoder3D(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        input_channels, base_features = config['input_channels'], config['base_features']
        self.config = config
        
        self.encoder1 = EncoderBlock(input_channels, base_features)
        self.encoder2 = EncoderBlock(base_features, base_features*2)
        self.encoder3 = EncoderBlock(base_features*2, base_features*4)
        self.encoder4 = EncoderBlock(base_features*4, base_features*8)
        self.encoder5 = EncoderBlock(base_features*8, base_features*16)

    def forward(self, x):
        x, f1 = self.encoder1(x) 
        x, f2 = self.encoder2(x)  
        x, f3 = self.encoder3(x)  
        x, f4 = self.encoder4(x)  
        x, f5 = self.encoder5(x)
        encs = [f1, f2, f3, f4, f5]
        return x, encs

class DecoderBlock(nn.Module):
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, x, skip_connections):
        x = self.upsample(x)
        x = self.conv_block(torch.cat([x, skip_connections], 1))
        return x


class Decoder3D(pl.LightningModule): 
    def __init__(self, config):
        super().__init__()
        base_features = config['base_features']
        self.middle = nn.Conv3d(base_features*16, base_features*16, kernel_size=3, padding=1) 
        self.bn_middle = nn.InstanceNorm3d(base_features*16)
        self.relu = nn.ReLU(inplace=True)

        self.decoder1 = DecoderBlock(base_features*16*2, base_features*8)
        self.decoder2 = DecoderBlock(base_features*8*2, base_features*4)
        self.decoder3 = DecoderBlock(base_features*4*2, base_features*2)
        self.decoder4 = DecoderBlock(base_features*2*2, base_features)
        self.decoder5 = DecoderBlock(base_features*2*1, base_features)

        self.last = nn.Conv3d(base_features, config['output_dim'], kernel_size=3, padding=1)

    def forward(self, x, features, get_features=False): 
        [f1, f2, f3, f4, f5] = features
        dm = self.relu(self.bn_middle(self.middle(x)))
        df1 = self.decoder1(dm, f5)  
        df2 = self.decoder2(df1, f4)  
        df3 = self.decoder3(df2, f3)  
        df4 = self.decoder4(df3, f2) 
        df5 = self.decoder5(df4, f1) 
        x = self.last(df5)
        if get_features:
            return x, [df1, df2, df3, df4, df5] 
        else:
            return x

class Unet3D(pl.LightningModule):
    def __init__(self, config):
        super(Unet3D, self).__init__()
        self.Enc = Encoder3D(config)   
        self.Dec = Decoder3D(config)

    def forward(self, x): 
        x, feature = self.Enc(x)
        x = self.Dec(x, feature)
        return x
      

