import torch
import clip
from PIL import Image
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance


def func_clip_T_score(image_path, text, model, preprocess, device):
    """Calcula la similitud coseno entre el embedding de una imagen y su prompt textual."""
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_input = clip.tokenize([text], truncate=True).to(device)
    model = model.to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return torch.matmul(image_features, text_features.T).item()


def func_clip_I_score(image_gen_path, image_ref_path, model, preprocess, device):
    """Calcula la similitud visual CLIP entre una imagen generada y una imagen de referencia."""
    image_gen = Image.open(image_gen_path).convert("RGB")
    image_ref = Image.open(image_ref_path).convert("RGB")

    image_gen_input = preprocess(image_gen).unsqueeze(0).to(device)
    image_ref_input = preprocess(image_ref).unsqueeze(0).to(device)
    model = model.to(device)

    with torch.no_grad():
        image_gen_features = model.encode_image(image_gen_input)
        image_ref_features = model.encode_image(image_ref_input)

    image_gen_features = image_gen_features / image_gen_features.norm(dim=-1, keepdim=True)
    image_ref_features = image_ref_features / image_ref_features.norm(dim=-1, keepdim=True)

    return torch.matmul(image_gen_features, image_ref_features.T).item()


def func_DINO_score(image_gen_path, image_ref_path, model, preprocess, device):
    """Calcula la similitud coseno entre representaciones extraídas con un modelo DINO."""
    image_gen = Image.open(image_gen_path).convert("RGB")
    image_ref = Image.open(image_ref_path).convert("RGB")

    image_gen_input = preprocess(image_gen).unsqueeze(0).to(device)
    image_ref_input = preprocess(image_ref).unsqueeze(0).to(device)

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        image_gen_features = model(image_gen_input)
        image_ref_features = model(image_ref_input)

    image_gen_features = F.normalize(image_gen_features, dim=-1)
    image_ref_features = F.normalize(image_ref_features, dim=-1)

    return torch.matmul(image_gen_features, image_ref_features.T).item()


def func_FID(real_img, fake_img, **fid_kwargs):
    """Wrapper para calcular Fréchet Inception Distance."""
    default_args = {
        "feature": 2048,
        "reset_real_features": True,
        "normalize": False,
        "input_img_size": (3, 299, 299),
        "feature_extractor_weights_path": None,
    }
    default_args.update(fid_kwargs)

    fid = FrechetInceptionDistance(**default_args)
    fid.update(real_img, real=True)
    fid.update(fake_img, real=False)
    return fid.compute()


def func_KID(real_img, fake_img, **kid_kwargs):
    """Wrapper para calcular Kernel Inception Distance."""
    default_args = {
        "feature": 2048,
        "reset_real_features": True,
        "normalize": False,
        "input_img_size": (3, 299, 299),
    }
    default_args.update(kid_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kid = KernelInceptionDistance(**default_args).to(device)
    kid.update(real_img, real=True)
    kid.update(fake_img, real=False)
    return kid.compute()
