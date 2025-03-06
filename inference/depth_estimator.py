import os, sys
sys.path.append(os.getcwd())  # 确保当前工作目录在sys.path中
# 添加以下行以确保models目录在sys.path中
sys.path.append(os.path.join(os.getcwd(), 'models', 'ml_depth_pro', 'src'))

from models.ml_depth_pro.src import depth_pro

class DepthEstimator:
    def __init__(self, model_name):
        self.model_name = model_name

        if self.model_name == 'ml-depth-pro':
            DEFAULT_MONODEPTH_CONFIG_DICT = depth_pro.DepthProConfig(patch_encoder_preset="dinov2l16_384",
                                                                     image_encoder_preset="dinov2l16_384",
                                                                     checkpoint_uri="/home/wangyike/Workspace/projects/awesome_depth_estimation/checkpoints/ml-depth-pro/depth_pro.pt",
                                                                     decoder_features=256,
                                                                     use_fov_head=True,
                                                                     fov_encoder_preset="dinov2l16_384")
            
            # Load model and preprocessing transform
            self.model, self.transform = depth_pro.create_model_and_transforms(DEFAULT_MONODEPTH_CONFIG_DICT)
            self.model.eval()        


    def inference(self, image_path):
        # Load and preprocess an image.
        image, _, f_px = depth_pro.load_rgb(image_path)
        image = self.transform(image)

        # Run inference.
        prediction = self.model.infer(image, f_px=f_px)
        depth_npy = prediction["depth"]  # Depth in [m].

        return depth_npy
    

if __name__ == '__main__':


    depth_estimator = DepthEstimator('ml-depth-pro')
    depth_npy = depth_estimator.inference("/mnt/nas/perception/yike/validation/temp/3_65/input_rgb_0/1737019012274192331.png")
    
    
