from .wan_video_vae import WanVideoVAE, WanVideoVAE38

class WanVideoVAELoader:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "loaders/wan_video" 

    def load_vae(self):
        vae = WanVideoVAE()
        return (vae,)

class WanVideoVAE38Loader:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae_38"
    CATEGORY = "loaders/wan_video"

    def load_vae_38(self):
        vae = WanVideoVAE38()
        return (vae,)

NODE_CLASS_MAPPINGS = {
    "WanVideoVAELoader": WanVideoVAELoader,
    "WanVideoVAE38Loader": WanVideoVAE38Loader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoVAELoader": "Load Wan Video VAE",
    "WanVideoVAE38Loader": "Load Wan Video VAE (38)",
}