import torch
import numpy as np
# Change imports for SigLIP
# from transformers import CLIPProcessor, CLIPModel
from transformers import AutoProcessor, AutoModel

class CLIPModelManager:
    """
    Classe singleton para gerenciar uma única instância do modelo Vision-Text (SigLIP),
    compartilhada entre os codificadores de imagem e texto.
    """
    _instance = None
    
    @classmethod
    # Update default model name to SigLIP
    def get_instance(cls, model_name="google/siglip2-so400m-patch16-512"):
        """
        Obtém a instância única do gerenciador de modelo.
        
        Args:
            model_name: Nome do modelo Vision-Text a ser carregado (Hugging Face Hub).
            
        Returns:
            Instância do gerenciador de modelo.
        """
        if cls._instance is None:
            cls._instance = cls(model_name)
        # Ensure the instance uses the requested model if already initialized with a different one
        elif cls._instance.model_name != model_name:
             print(f"Warning: Existing instance uses {cls._instance.model_name}. Reinitializing with {model_name}.")
             cls._instance = cls(model_name)
        return cls._instance
    
    def __init__(self, model_name):
        """
        Inicializa o gerenciador de modelo.
        
        Args:
            model_name: Nome do modelo a ser carregado.
        """
        # Use device_map="auto" for automatic device placement
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Vision-Text model '{model_name}' using device_map='auto'")
        
        # Load SigLIP model and processor
        self.model = AutoModel.from_pretrained(model_name, device_map="auto").eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model_name = model_name
        # Store the device the model is actually using
        self.device = self.model.device
        print(f"Model {model_name} loaded on device: {self.device}")
    
    def encode_image(self, image_inputs):
        """
        Codifica uma imagem em um embedding.
        
        Args:
            image_inputs: Imagem processada pelo processador.
            
        Returns:
            Embedding da imagem como um array numpy.
        """
        # Ensure inputs are on the correct device (redundant if handled in caller? Check caller)
        # image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
        with torch.no_grad():
            # Use get_image_features for SigLIP/CLIP models
            outputs = self.model.get_image_features(**image_inputs)
        
        # Normalizar o embedding
        embedding = outputs.cpu().numpy()[0]
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def encode_text(self, text_inputs):
        """
        Codifica um texto em um embedding.
        
        Args:
            text_inputs: Texto processado pelo processador.
            
        Returns:
            Embedding do texto como um array numpy.
        """
        # Ensure inputs are on the correct device
        # text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        with torch.no_grad():
            # Use get_text_features for SigLIP/CLIP models
            text_features = self.model.get_text_features(**text_inputs)
        
        # Converter para numpy e normalizar
        embedding = text_features.cpu().numpy()[0]
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def compute_similarity(self, embedding1, embedding2):
        """
        Calcula a similaridade de cosseno entre dois embeddings.
        
        Args:
            embedding1: Primeiro embedding.
            embedding2: Segundo embedding.
            
        Returns:
            Similaridade de cosseno entre os embeddings.
        """
        # Dot product of normalized embeddings is cosine similarity
        return np.dot(embedding1, embedding2) 