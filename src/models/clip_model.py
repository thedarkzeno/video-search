import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel

class CLIPModelManager:
    """
    Classe singleton para gerenciar uma única instância do modelo CLIP,
    compartilhada entre os codificadores de imagem e texto.
    """
    _instance = None
    
    @classmethod
    def get_instance(cls, model_name="adalbertojunior/clip-H-portuguese"):
        """
        Obtém a instância única do gerenciador de modelo CLIP.
        
        Args:
            model_name: Nome do modelo CLIP a ser carregado.
            
        Returns:
            Instância do gerenciador de modelo CLIP.
        """
        if cls._instance is None:
            cls._instance = cls(model_name)
        return cls._instance
    
    def __init__(self, model_name):
        """
        Inicializa o gerenciador de modelo CLIP.
        
        Args:
            model_name: Nome do modelo CLIP a ser carregado.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP model '{model_name}' on device: {self.device}")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model_name = model_name
    
    def encode_image(self, image):
        """
        Codifica uma imagem em um embedding.
        
        Args:
            image: Imagem processada pelo processador CLIP.
            
        Returns:
            Embedding da imagem como um array numpy.
        """
        with torch.no_grad():
            outputs = self.model.get_image_features(**image)
        
        # Normalizar o embedding
        embedding = outputs.cpu().numpy()[0]
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def encode_text(self, text_inputs):
        """
        Codifica um texto em um embedding.
        
        Args:
            text_inputs: Texto processado pelo processador CLIP.
            
        Returns:
            Embedding do texto como um array numpy.
        """
        with torch.no_grad():
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
        return np.dot(embedding1, embedding2) 