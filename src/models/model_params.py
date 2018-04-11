#============ Basic models ============#
from models.LinkNet import LinkNet34,LinkNet50,LinkNeXt

#============ Inception based models ============#
from models.LinkNet import LinkInceptionResNet,LinkCeption

#============ Heavier models ============#
from models.LinkNet import ResNet34Unet
from models.VggUNet import UNet11
from models.GCN import FCN
from models.TernausNet import AlbuNet,UNet16,UNet16Mod

model_presets = {
    'albunet_3' : [AlbuNet,{'num_classes':3,'is_deconv':False,'pretrained':True,'num_filters':32}],
    'unet16_3' : [UNet16,{'num_classes':3,'is_deconv':False,'pretrained':True,'num_filters':32}],
    
    'albunet_3_zero' : [AlbuNet,{'num_classes':3,'is_deconv':False,'pretrained':False,'num_filters':32}],
    'unet16_3_zero' : [UNet16,{'num_classes':3,'is_deconv':False,'pretrained':False,'num_filters':32}],    
    
    'unet11_3' : [UNet11,{'num_classes':3,'num_channels':3}],
    'linknet50_3': [LinkNet50,{'num_classes':3,'num_channels':3,'is_deconv':False,'decoder_kernel_size':3,}],
    'resunet34_3': [ResNet34Unet,{'num_classes':3,'num_channels':3,'is_deconv':False,'decoder_kernel_size':3,}],    

    'unet16_3_dc' : [UNet16,{'num_classes':3,'is_deconv':True,'pretrained':True,'num_filters':32}],
    
    'unet16_5' : [UNet16,{'num_classes':5,'is_deconv':False,'pretrained':True,'num_filters':32}],    
    'unet16_5_dc' : [UNet16,{'num_classes':5,'is_deconv':True,'pretrained':True,'num_filters':32}],
    'unet16_6_dc' : [UNet16,{'num_classes':6,'is_deconv':True,'pretrained':True,'num_filters':32}],
    
    'unet16_6_dc' : [UNet16,{'num_classes':6,'is_deconv':True,'pretrained':True,'num_filters':32}],
    'unet16_64_6_dc' : [UNet16,{'num_classes':6,'is_deconv':True,'pretrained':True,'num_filters':64}],
    
    'unet16_128_6_dc' : [UNet16,{'num_classes':6,'is_deconv':True,'pretrained':True,'num_filters':128}],
    
    'unet16_16_7_dc' : [UNet16,{'num_classes':7,'is_deconv':True,'pretrained':True,'num_filters':16}], 
    'unet16_32_7_dc' : [UNet16,{'num_classes':7,'is_deconv':True,'pretrained':True,'num_filters':32}],      
    'unet16_160_7_dc' : [UNet16,{'num_classes':7,'is_deconv':True,'pretrained':True,'num_filters':160}],
    'unet16_128_7_dc' : [UNet16,{'num_classes':7,'is_deconv':True,'pretrained':True,'num_filters':128}],
    
    'unet16mod_64_7_dc' : [UNet16Mod,{'num_classes':7,'is_deconv':True,'pretrained':True,'num_filters':64}],
    
    'unet16_64_7_dc' : [UNet16,{'num_classes':7,'is_deconv':True,'pretrained':True,'num_filters':64}],
    'unet16_64_8_dc' : [UNet16,{'num_classes':8,'is_deconv':True,'pretrained':True,'num_filters':64}],      
    
    'albunet32_6_dc' : [AlbuNet,{'num_classes':6,'is_deconv':True,'pretrained':True,'num_filters':32}],
    'albunet64_6_dc' : [AlbuNet,{'num_classes':6,'is_deconv':True,'pretrained':True,'num_filters':64}],       
    
    'albunet_3_dc' : [AlbuNet,{'num_classes':3,'is_deconv':True,'pretrained':True,'num_filters':32}],
    'unet11_3_dc' : [UNet11,{'num_classes':3,'num_channels':3,'is_deconv':True}],
    'linknet50_3_dc': [LinkNet50,{'num_classes':3,'num_channels':3,'is_deconv':True,'decoder_kernel_size':3,}],
    'resunet34_3_dc': [ResNet34Unet,{'num_classes':3,'num_channels':3,'is_deconv':True,'decoder_kernel_size':3,}],           
    
    
    'linknext_3': [LinkNeXt,{'num_classes':3,'num_channels':3,'upsampling':'bu','decoder_kernel_size':3,}],
    'linkception_3': [LinkCeption,{'num_classes':3,'num_channels':3,'upsampling':'bu','decoder_kernel_size':3,}],    
    'linkceptionresnet_3': [LinkInceptionResNet,{'num_classes':3,'num_channels':3,'upsampling':'bu','decoder_kernel_size':3,}],
    'gcn50_3' : [FCN,{'num_classes':3}],    

}