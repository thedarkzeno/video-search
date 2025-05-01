import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from .clip_model import CLIPModelManager

class ImageEncoder:
    """
    Classe para codificar imagens em embeddings usando o modelo CLIP.
    """
    def __init__(self):
        # Usar a instância compartilhada do modelo CLIP
        self.clip_manager = CLIPModelManager.get_instance()
        # Update print message to reflect the potentially different model
        print(f"ImageEncoder using shared model: {self.clip_manager.model_name} on device {self.clip_manager.device}")
    
    def encode_image(self, image):
        """
        Codifica uma imagem em um embedding.
        
        Args:
            image: Imagem PIL ou caminho para a imagem.
            
        Returns:
            Embedding da imagem como um array numpy.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        # Processar a imagem
        # Move inputs to the device the model manager determined
        inputs = self.clip_manager.processor(images=image, return_tensors="pt").to(self.clip_manager.device)
        
        # Usar o gerenciador para codificar
        return self.clip_manager.encode_image(inputs)
    
    def compute_similarity(self, embedding1, embedding2):
        """
        Calcula a similaridade de cosseno entre dois embeddings.
        
        Args:
            embedding1: Primeiro embedding.
            embedding2: Segundo embedding.
            
        Returns:
            Similaridade de cosseno entre os embeddings.
        """
        return self.clip_manager.compute_similarity(embedding1, embedding2)


class ImageCaptioner:
    """
    Classe para gerar descrições de imagens.
    """
    def __init__(self, model_name="VerboVision/VerboVision-base"):
        from transformers import AutoProcessor, LlavaNextForConditionalGeneration
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device)
    
    # def generate_caption(self, image):
    #     """
    #     Gera uma descrição para uma imagem.
        
    #     Args:
    #         image: Imagem PIL ou caminho para a imagem.
            
    #     Returns:
    #         Descrição da imagem.
    #     """
    #     if isinstance(image, str):
    #         image = Image.open(image).convert("RGB")
        
    #     inputs = self.processor(image, return_tensors="pt").to(self.device)
        
    #     with torch.no_grad():
    #         outputs = self.model.generate(**inputs, max_length=50)
        
    #     caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        
    #     return caption 
    def generate_caption(self, image):
        # Define a chat histiry and use apply_chat_template to get correctly formatted prompt
        # Each value in "content" has to be a list of dicts with types ("text", "image") 
        conversation = [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Descreva brevemente a imagem a seguir."},
                ],
            },
        ]
        prompts = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        inputs = self.processor(images=[image], text=[prompts], return_tensors='pt').to(0, torch.float16)
        
        output = self.model.generate(**inputs, max_new_tokens=256, do_sample=False, temperature=0.5)
        torch.cuda.empty_cache()
        return (self.processor.decode(output[0], skip_special_tokens=True)).split("assistant\n\n")[-1]