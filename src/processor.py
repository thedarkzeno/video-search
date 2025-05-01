import os
import cv2
import numpy as np
import tempfile
from pathlib import Path
from datetime import datetime
import time  # Import the time module

from src.database.db import DBManager
from src.models.image_models import ImageEncoder, ImageCaptioner
from src.models.text_models import AudioTranscriber, TextEncoder

class VideoProcessor:
    """
    Classe para processar vídeos, extrair frames principais, gerar descrições e transcrever áudio.
    """
    def __init__(self, 
                 db_path=None,
                 index_path=None,
                 frame_similarity_threshold=0.8,
                 frame_interval=1.0,
                 output_dir="processed_data"):
        """
        Inicializa o processador de vídeos.
        
        Args:
            db_path: Caminho para o banco de dados SQLite. Se None, será usado "video_search.db" no output_dir.
            index_path: Caminho para o índice FAISS. Se None, será usado "embeddings_index" no output_dir.
            frame_similarity_threshold: Limiar de similaridade para considerar um frame como principal.
            frame_interval: Intervalo em segundos para extrair frames do vídeo.
            output_dir: Diretório para salvar frames e áudios extraídos.
        """
        self.output_dir = output_dir
        
        # Criar diretórios de saída
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Configurar caminhos para o banco de dados e índice
        if db_path is None:
            db_path = os.path.join(output_dir, "video_search.db")
        if index_path is None:
            index_path = os.path.join(output_dir, "embeddings_index")
        
        # Inicializar o gerenciador de banco de dados
        self.db_manager = DBManager(db_path=db_path, index_path=index_path)
        
        self.frame_similarity_threshold = frame_similarity_threshold
        self.frame_interval = frame_interval
        
        # Criar diretórios de saída para frames e áudio
        self.frames_dir = os.path.join(output_dir, "frames")
        self.audio_dir = os.path.join(output_dir, "audio")
        Path(self.frames_dir).mkdir(parents=True, exist_ok=True)
        Path(self.audio_dir).mkdir(parents=True, exist_ok=True)
        
        # Inicializar modelos
        self.image_encoder = ImageEncoder()
        self.image_captioner = ImageCaptioner()
        self.audio_transcriber = AudioTranscriber()
        self.text_encoder = TextEncoder()
    
    def process_video(self, video_path):
        """
        Processa um vídeo completo: extrai frames principais, gera descrições e transcreve áudio.
        Calcula e armazena o tempo de processamento.
        
        Args:
            video_path: Caminho para o arquivo de vídeo.
            
        Returns:
            ID do vídeo no banco de dados.
        """
        start_time = time.time()  # Record start time
        print(f"Iniciando processamento do vídeo: {video_path}")
        
        video_id = None # Initialize video_id
        try:
            # Verificar se o vídeo existe
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")
            
            # Abrir o vídeo
            video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")
            
            # Obter informações do vídeo
            fps = video.get(cv2.CAP_PROP_FPS)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            # Handle potential division by zero or invalid fps
            duration = frame_count / fps if fps > 0 else 0 
            
            # Adicionar vídeo ao banco de dados (sem tempo de processamento ainda)
            video_id = self.db_manager.add_video(video_path, duration)
            if video_id is None: # Check if video was added successfully
                 raise Exception("Failed to add video to database.")
            
            # --- Limpar dados antigos antes de reprocessar ---
            print(f"Cleaning up previous data for video ID: {video_id}...")
            self.db_manager.delete_key_frames_for_video(video_id)
            self.db_manager.delete_audio_transcriptions_for_video(video_id)
            # Note: Associated frame files and audio file are not deleted here, only DB entries.
            # Note: FAISS index entries are not removed by the current DB methods.
            # ------------------------------------------------
            
            # Extrair frames principais (medir tempo)
            frame_start_time = time.time()
            key_frames = self._extract_key_frames(video, video_id, fps)
            frame_time = time.time() - frame_start_time
            print(f"Frame extraction/description took {frame_time:.2f} seconds.")
            
            # Extrair e transcrever áudio (medir tempo)
            audio_start_time = time.time()
            self._extract_and_transcribe_audio(video_path, video_id)
            audio_time = time.time() - audio_start_time
            print(f"Audio extraction/transcription took {audio_time:.2f} seconds.")
            
            video.release()

            # Calculate total processing time 
            total_processing_time = time.time() - start_time
            print(f"Processamento total concluído em {total_processing_time:.2f} segundos. Frames principais: {len(key_frames)}")
            
            # Atualizar os tempos de processamento no banco de dados
            self.db_manager.update_video_processing_time(video_id, total_processing_time, frame_time, audio_time)
            
            return video_id
            
        except Exception as e:
             # If an error occurs after video_id is assigned, maybe log it? 
             # For now, just print and re-raise or handle as appropriate
             print(f"Erro durante o processamento do vídeo {video_path}: {e}")
             # Optionally cleanup partial data or re-raise the exception
             # raise e 
             return None # Or indicate failure differently
    
    def get_db_path(self):
        """
        Retorna o caminho para o banco de dados SQLite.
        
        Returns:
            Caminho para o banco de dados.
        """
        return self.db_manager.db_path
    
    def get_index_path(self):
        """
        Retorna o caminho para o índice FAISS.
        
        Returns:
            Caminho para o índice.
        """
        return self.db_manager.index_path
    
    def _extract_key_frames(self, video, video_id, fps):
        """
        Extrai frames principais do vídeo com base na similaridade.
        
        Args:
            video: Objeto VideoCapture do OpenCV.
            video_id: ID do vídeo no banco de dados.
            fps: Frames por segundo do vídeo.
            
        Returns:
            Lista de IDs dos frames principais.
        """
        key_frames = []
        last_key_frame_embedding = None
        frame_interval_steps = int(self.frame_interval * fps)
        
        frame_index = 0
        while True:
            # Definir a posição do frame
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            
            # Ler o frame
            ret, frame = video.read()
            if not ret:
                break
            
            # Calcular o timestamp
            timestamp = frame_index / fps
            
            # Converter BGR para RGB (para compatibilidade com os modelos)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Codificar o frame
            current_embedding = self.image_encoder.encode_image(Image.fromarray(frame_rgb))
            
            # Verificar se é um frame principal
            is_key_frame = False
            if last_key_frame_embedding is None:
                # Primeiro frame é sempre um frame principal
                is_key_frame = True
            else:
                # Calcular similaridade com o último frame principal
                similarity = self.image_encoder.compute_similarity(last_key_frame_embedding, current_embedding)
                is_key_frame = similarity < self.frame_similarity_threshold
            
            if is_key_frame:
                # Salvar o frame
                frame_filename = f"video_{video_id}_frame_{frame_index}_{timestamp:.2f}.jpg"
                frame_path = os.path.join(self.frames_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                
                # Gerar descrição
                description = self.image_captioner.generate_caption(Image.fromarray(frame_rgb))
                
                # Adicionar ao banco de dados
                frame_id = self.db_manager.add_key_frame(video_id, timestamp, frame_path, description)
                
                # Adicionar embedding ao índice FAISS
                self.db_manager.add_frame_embedding(frame_id, current_embedding)
                
                # Atualizar último frame principal
                last_key_frame_embedding = current_embedding
                key_frames.append(frame_id)
                
                print(f"Frame principal encontrado em {timestamp:.2f}s: {description}")
            
            # Avançar para o próximo frame a ser verificado
            frame_index += frame_interval_steps
        
        return key_frames
    
    def _extract_and_transcribe_audio(self, video_path, video_id):
        """
        Extrai e transcreve o áudio do vídeo.
        
        Args:
            video_path: Caminho para o arquivo de vídeo.
            video_id: ID do vídeo no banco de dados.
        """
        # Extrair áudio para um arquivo temporário
        audio_filename = f"video_{video_id}_audio.wav"
        audio_path = os.path.join(self.audio_dir, audio_filename)
        
        # Usar ffmpeg para extrair o áudio
        os.system(f'ffmpeg -i "{video_path}" -q:a 0 -map a "{audio_path}" -y')
        
        if not os.path.exists(audio_path):
            print(f"Aviso: Não foi possível extrair áudio de {video_path}")
            return
        
        # Transcrever áudio em segmentos
        segments = self.audio_transcriber.transcribe_audio_segments(audio_path)
        
        # Adicionar transcrições ao banco de dados
        for segment in segments:
            start_time = segment["start"]
            end_time = segment["end"]
            text = segment["text"]
            
            self.db_manager.add_audio_transcription(video_id, start_time, end_time, text)
            
            # Opcional: Codificar o texto e armazenar o embedding
            # text_embedding = self.text_encoder.encode_text(text)
            # Aqui você poderia armazenar o embedding do texto se necessário
            
        print(f"Áudio transcrito: {len(segments)} segmentos")

# Importação necessária para PIL.Image
from PIL import Image
