from .models import HYVideoDiffusionTransformer, HUNYUAN_VIDEO_CONFIG

def load_model(args, in_channels, out_channels, video_condition, audio_condition, factor_kwargs):
    model = HYVideoDiffusionTransformer(
        args,
        in_channels=in_channels,
        out_channels=out_channels,
        video_condition=video_condition,
        audio_condition=audio_condition,
        **HUNYUAN_VIDEO_CONFIG[args.model],
        **factor_kwargs,
    )
    return model
