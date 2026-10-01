import torch
import clip
from PIL import Image
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance



def func_clip_T_score(image_path, text, model, preprocess, device):

    # Load the pre-trained CLIP model and the image
    #model, preprocess = clip.load('ViT-B/32')
    image = Image.open(image_path).convert("RGB")

    # Preprocess the image and tokenize the text
    image_input = preprocess(image).unsqueeze(0)
    text_input = clip.tokenize([text])

    # Move the inputs to GPU if available
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    image_input = image_input.to(device)
    text_input = text_input.to(device)
    model = model.to(device)

    # Generate embeddings for the image and text
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)

    # Normalize the features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Calculate the cosine similarity to get the CLIP score
    clip_score = torch.matmul(image_features, text_features.T).item()

    return clip_score


def func_clip_I_score(image_gen_path, image_ref_path, model, preprocess, device):
    # Load the pre-trained CLIP model and the image
    #model, preprocess = clip.load('ViT-B/32')
    image_gen = Image.open(image_gen_path).convert("RGB")
    image_ref = Image.open(image_ref_path).convert("RGB")

    # Preprocess the image and tokenize the text
    image_gen_input = preprocess(image_gen).unsqueeze(0)
    image_ref_input = preprocess(image_ref).unsqueeze(0)


    # Move the inputs to GPU if available
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    image_gen_input = image_gen_input.to(device)
    image_ref_input = image_ref_input.to(device)
    model = model.to(device)

    # Generate embeddings for the image and text
    with torch.no_grad():
        image_gen_features = model.encode_image(image_gen_input)
        image_ref_features = model.encode_image(image_ref_input)


    # Normalize the features
    image_gen_features = image_gen_features / image_gen_features.norm(dim=-1, keepdim=True)
    image_ref_features = image_ref_features / image_ref_features.norm(dim=-1, keepdim=True)


    # Calculate the cosine similarity to get the CLIP score
    clip_score = torch.matmul(image_gen_features, image_ref_features.T).item()

    return clip_score

def func_DINO_score(image_gen_path, image_ref_path, model, preprocess, device):
    # Cargar y procesar las imágenes
    image_gen = Image.open(image_gen_path).convert('RGB')
    image_ref = Image.open(image_ref_path).convert('RGB')

    image_gen_input = preprocess(image_gen).unsqueeze(0).to(device)
    image_ref_input = preprocess(image_ref).unsqueeze(0).to(device)

    # Enviar modelo a dispositivo
    model = model.to(device)
    model.eval()

    # Obtener features DINO
    with torch.no_grad():
        image_gen_features = model(image_gen_input)
        image_ref_features = model(image_ref_input)

    # Normalizar features
    image_gen_features = F.normalize(image_gen_features, dim=-1)
    image_ref_features = F.normalize(image_ref_features, dim=-1)

    # Similaridad coseno
    dino_score = torch.matmul(image_gen_features, image_ref_features.T).item()

    return dino_score

def func_FID(real_img, fake_img, **fid_kwargs):

    # Valores por defecto si no se proporcionan
    default_args = {
        "feature": 2048,
        "reset_real_features": True,
        "normalize": False,
        "input_img_size": (3, 299, 299),
        "feature_extractor_weights_path": None
    }

    # Actualizar los valores por defecto con los argumentos proporcionados
    default_args.update(fid_kwargs)

    # Crear la instancia con los parámetros actualizados
    fid = FrechetInceptionDistance(**default_args)

    # Calcular el FID
    fid.update(real_img, real=True)
    fid.update(fake_img, real=False)
    return fid.compute()

def func_KID(real_img, fake_img, **kid_kwargs):

    # Valores por defecto si no se proporcionan
    default_args = {
        "feature": 2048,
        "reset_real_features": True,
        "normalize": False,
        "input_img_size": (3, 299, 299),
    }

    # Actualizar los valores por defecto con los argumentos proporcionados
    default_args.update(kid_kwargs)

    # Crear la instancia con los parámetros actualizados
    kid = KernelInceptionDistance(**default_args)

    # Calcular el KID
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kid = kid.to(device)
    kid.update(real_img, real=True)
    kid.update(fake_img, real=False)
    return kid.compute()




