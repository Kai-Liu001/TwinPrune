"""InternVL loading and the exact main-table image preprocessing."""

import torch

DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def load_internvl(model_name: str = "OpenGVLab/InternVL3_5-8B-Instruct", device: str = "cuda:0",
                 dtype: str = "bfloat16", device_map: str | None = None):
    """Load InternVL model, tokenizer, and processor."""
    from transformers import AutoModel, AutoTokenizer

    path = model_name
    from src.internvl_compat import prepare_checkpoint_class
    prepare_checkpoint_class(path)
    target_dtype = DTYPE_MAP.get(dtype, torch.bfloat16)
    placement = {'device_map': device_map} if device_map else {}
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=target_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=bool(device_map),
        **placement,
    )
    if not device_map:
        model = model.to(device)
    model = model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, use_fast=True,
    )

    # Add InternVL special tokens if missing
    special_tokens = ["<IMG_CONTEXT>", "<img>", "</img>"]
    tokens_to_add = [t for t in special_tokens if tokenizer.convert_tokens_to_ids(t) is None]
    if tokens_to_add:
        tokenizer.add_tokens(tokens_to_add, special_tokens=True)
        model.language_model.resize_token_embeddings(len(tokenizer))

    # Set img_context_token_id on model
    img_ctx_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    model.img_context_token_id = img_ctx_id

    from src.internvl_conversation import register_model
    register_model(model)

    return model, tokenizer


def build_image_transform(input_size=448):
    """Build standard InternVL image transform."""
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def prepare_pixel_values(image, model, device: str = "cuda:0", input_size: int = 448) -> torch.Tensor:
    """Prepare pixel_values tensor from a PIL Image."""
    transform = build_image_transform(input_size)
    return transform(image).unsqueeze(0).to(device=device, dtype=model.dtype)
