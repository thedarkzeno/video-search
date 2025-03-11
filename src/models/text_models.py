import os
import torch
import whisper
import numpy as np
from sentence_transformers import SentenceTransformer
from .clip_model import CLIPModelManager

class AudioTranscriber:
    """
    Classe para transcrever áudio usando o modelo Whisper.
    """
    def __init__(self, model_size="base"):
        """
        Inicializa o transcritor de áudio.
        
        Args:
            model_size: Tamanho do modelo Whisper ("tiny", "base", "small", "medium", "large").
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device} for Whisper")
        
        self.model = whisper.load_model(model_size).to(self.device)
    
    def transcribe_audio(self, audio_path):
        """
        Transcreve um arquivo de áudio.
        
        Args:
            audio_path: Caminho para o arquivo de áudio.
            
        Returns:
            Texto transcrito.
        """
        result = self.model.transcribe(audio_path)
        return result["text"]
    
    def transcribe_audio_segments(self, audio_path, segment_length=30):
        """
        Transcreve um arquivo de áudio em segmentos.
        
        Args:
            audio_path: Caminho para o arquivo de áudio.
            segment_length: Duração de cada segmento em segundos.
            
        Returns:
            Lista de segmentos transcritos com timestamps.
        """
        result = self.model.transcribe(audio_path, word_timestamps=True)
        segments = []
        
        for segment in result["segments"]:
            segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            })
        
        return segments


class TextEncoder:
    """
    Classe para codificar texto em embeddings usando o mesmo modelo CLIP usado para imagens.
    Isso garante que os embeddings de texto e imagem estejam no mesmo espaço vetorial.
    """
    def __init__(self):
        """
        Inicializa o codificador de texto usando o modelo CLIP compartilhado.
        
        Args:
            model_name: Nome do modelo CLIP.
        """
        # Usar a instância compartilhada do modelo CLIP
        self.clip_manager = CLIPModelManager.get_instance()
        print(f"TextEncoder usando modelo CLIP compartilhado: {self.clip_manager.model_name}")
    
    def encode_text(self, text):
        """
        Codifica um texto em um embedding usando o codificador de texto do CLIP.
        
        Args:
            text: Texto a ser codificado.
            
        Returns:
            Embedding do texto como um array numpy.
        """
        # Processar o texto usando o processador CLIP
        inputs = self.clip_manager.processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.clip_manager.device)
        
        # Usar o gerenciador para codificar
        return self.clip_manager.encode_text(inputs)
    
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