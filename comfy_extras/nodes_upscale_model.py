import os
from comfy_extras.chainner_models import model_loading
from comfy.sd import load_torch_file
import comfy.model_management
from nodes import filter_files_extensions, recursive_search, supported_ckpt_extensions
import torch
import comfy.utils
import configparser


# 获取当前脚本的路径
current_script_path = os.path.dirname(os.path.realpath(__file__))

# 获取上一层目录的路径
parent_directory_path = os.path.dirname(current_script_path)

# 构建配置文件的完整路径
config_file_path = os.path.join(parent_directory_path, 'model_path.ini')

myconfig = configparser.ConfigParser()
myconfig.read(config_file_path, encoding="utf-8")

print("Available sections:", myconfig.sections())
models_paths = myconfig.get('model_path', 'paths')

class UpscaleModelLoader:
    models_dir = models_paths
    upscale_model_dir = os.path.join(models_dir, "upscale_models")

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model_name": (filter_files_extensions(recursive_search(s.upscale_model_dir), supported_ckpt_extensions), ),
                             }}
    RETURN_TYPES = ("UPSCALE_MODEL",)
    FUNCTION = "load_model"

    CATEGORY = "loaders"

    def load_model(self, model_name):
        model_path = os.path.join(self.upscale_model_dir, model_name)
        sd = load_torch_file(model_path)
        out = model_loading.load_state_dict(sd).eval()
        return (out, )


class ImageUpscaleWithModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "upscale_model": ("UPSCALE_MODEL",),
                              "image": ("IMAGE",),
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "image/upscaling"

    def upscale(self, upscale_model, image):
        device = comfy.model_management.get_torch_device()
        upscale_model.to(device)
        in_img = image.movedim(-1,-3).to(device)
        s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=128 + 64, tile_y=128 + 64, overlap = 8, upscale_amount=upscale_model.scale)
        upscale_model.cpu()
        s = torch.clamp(s.movedim(-3,-1), min=0, max=1.0)
        return (s,)

NODE_CLASS_MAPPINGS = {
    "UpscaleModelLoader": UpscaleModelLoader,
    "ImageUpscaleWithModel": ImageUpscaleWithModel
}
