import os
import numpy as np
from PIL import Image

from src.database.db import DBManager
from src.models.image_models import ImageEncoder
from src.models.text_models import TextEncoder

class VideoSearch:
    """
    Classe para realizar buscas em vídeos processados.
    """
    def __init__(self, db_path=None, index_path=None, output_dir="processed_data"):
        """
        Inicializa o mecanismo de busca de vídeos.
        
        Args:
            db_path: Caminho para o banco de dados SQLite. Se None, será usado "video_search.db" no output_dir.
            index_path: Caminho para o índice FAISS. Se None, será usado "embeddings_index" no output_dir.
            output_dir: Diretório onde estão os dados processados.
        """
        # Configurar caminhos para o banco de dados e índice
        if db_path is None:
            db_path = os.path.join(output_dir, "video_search.db")
        if index_path is None:
            index_path = os.path.join(output_dir, "embeddings_index")
        
        # Inicializar o gerenciador de banco de dados
        self.db_manager = DBManager(db_path=db_path, index_path=index_path)
        
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
    
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
    
    def search_by_text(self, query_text, limit=10, semantic_weight=0.5):
        """
        Busca vídeos por texto, procurando nas descrições de frames e transcrições de áudio.
        Combina busca textual (correspondências parciais) com busca semântica (embeddings).
        
        Args:
            query_text: Texto de consulta.
            limit: Número máximo de resultados.
            semantic_weight: Peso para a busca semântica (0-1). Padrão: 0.5
            
        Returns:
            Lista de resultados ordenados por relevância combinada.
        """
        print(f"Buscando por texto: '{query_text}'")
        
        # Gerar embedding do texto de consulta
        query_embedding = self.text_encoder.encode_text(query_text)
        print(query_embedding.shape)
        
        # Buscar no banco de dados com correspondência parcial e semântica
        results = self.db_manager.search_by_text(
            query_text, 
            embedding=query_embedding,
            semantic_weight=semantic_weight
        )
        
        # Organizar resultados
        formatted_results = self._format_results(results, limit)
        
        return formatted_results
    
    def search_by_image(self, image_path, limit=10):
        """
        Busca vídeos por similaridade de imagem.
        
        Args:
            image_path: Caminho para a imagem de consulta.
            limit: Número máximo de resultados.
            
        Returns:
            Lista de resultados ordenados por similaridade.
        """
        print(f"Buscando por imagem: '{image_path}'")
        
        # Verificar se a imagem existe
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
        
        # Codificar a imagem
        query_image = Image.open(image_path).convert("RGB")
        query_embedding = self.image_encoder.encode_image(query_image)
        
        # Buscar por similaridade
        results = self.db_manager.search_by_embedding(query_embedding, top_k=limit)
        
        # Organizar resultados
        formatted_results = self._format_results(results, limit)
        
        return formatted_results
    
    def search_by_embedding(self, query_embedding, limit=10):
        """
        Busca vídeos por embedding diretamente.
        
        Args:
            query_embedding: Embedding de consulta.
            limit: Número máximo de resultados.
            
        Returns:
            Lista de resultados ordenados por similaridade.
        """
        # Buscar por similaridade
        results = self.db_manager.search_by_embedding(query_embedding, top_k=limit)
        
        # Organizar resultados
        formatted_results = self._format_results(results, limit)
        
        return formatted_results
    
    def search_multimodal(self, query_text=None, query_image=None, weights=(0.5, 0.5), limit=10):
        """
        Realiza uma busca multimodal combinando texto e imagem.
        
        Args:
            query_text: Texto de consulta (opcional).
            query_image: Caminho para a imagem de consulta (opcional).
            weights: Pesos para combinar os resultados (texto, imagem).
            limit: Número máximo de resultados.
            
        Returns:
            Lista de resultados ordenados por relevância combinada.
        """
        results_text = []
        results_image = []
        
        # Buscar por texto se fornecido
        if query_text:
            results_text = self.search_by_text(query_text, limit=limit*2)
        
        # Buscar por imagem se fornecida
        if query_image:
            results_image = self.search_by_image(query_image, limit=limit*2)
        
        # Se apenas um tipo de busca foi realizado, retornar seus resultados
        if not results_text:
            return results_image[:limit]
        if not results_image:
            return results_text[:limit]
        
        # Combinar resultados
        combined_results = self._combine_results(results_text, results_image, weights, limit)
        
        return combined_results
    
    def _format_results(self, results, limit):
        """
        Formata os resultados da busca.
        
        Args:
            results: Resultados brutos da busca.
            limit: Número máximo de resultados.
            
        Returns:
            Lista formatada de resultados.
        """
        formatted_results = []
        
        for result in results[:limit]:
            # Obter o match_type do resultado original
            match_type = result.get('match_type', 'unknown')
            score = result.get('score', 0.0)
            
            if 'frame_id' in result:
                # Resultado de busca por imagem
                frame_id = result['frame_id']
                
                # Obter informações do frame
                conn = self.db_manager._get_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT kf.video_id, kf.timestamp, kf.frame_path, kf.description, v.file_path
                    FROM key_frames kf
                    JOIN videos v ON kf.video_id = v.id
                    WHERE kf.id = ?
                """, (frame_id,))
                
                row = cursor.fetchone()
                conn.close()
                
                if row:
                    video_id, timestamp, frame_path, description, video_path = row
                    
                    formatted_results.append({
                        'type': 'frame',
                        'video_id': video_id,
                        'video_path': video_path,
                        'frame_id': frame_id,
                        'timestamp': timestamp,
                        'frame_path': frame_path,
                        'description': description,
                        'score': score,
                        'match_type': match_type
                    })
            
            elif 'transcription_id' in result:
                # Resultado de busca por texto/transcrição
                transcription_id = result['transcription_id']
                
                # Obter informações da transcrição
                conn = self.db_manager._get_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT at.video_id, at.start_time, at.end_time, at.text, v.file_path
                    FROM audio_transcriptions at
                    JOIN videos v ON at.video_id = v.id
                    WHERE at.id = ?
                """, (transcription_id,))
                
                row = cursor.fetchone()
                conn.close()
                
                if row:
                    video_id, start_time, end_time, text, video_path = row
                    
                    formatted_results.append({
                        'type': 'transcription',
                        'video_id': video_id,
                        'video_path': video_path,
                        'transcription_id': transcription_id,
                        'start_time': start_time,
                        'end_time': end_time,
                        'text': text,
                        'score': score,
                        'match_type': match_type
                    })
        
        return formatted_results
    
    def _combine_results(self, results_text, results_image, weights, limit):
        """
        Combina resultados de diferentes tipos de busca.
        
        Args:
            results_text: Resultados da busca por texto.
            results_image: Resultados da busca por imagem.
            weights: Pesos para combinar os resultados (texto, imagem).
            limit: Número máximo de resultados.
            
        Returns:
            Lista combinada de resultados.
        """
        # Normalizar pesos
        weight_sum = sum(weights)
        weights = tuple(w / weight_sum for w in weights)
        
        # Criar dicionário para combinar resultados por vídeo
        video_scores = {}
        
        # Adicionar resultados de texto
        for i, result in enumerate(results_text):
            video_id = result['video_id']
            # Pontuação inversa à posição no ranking, normalizada
            score = (len(results_text) - i) / len(results_text) * weights[0]
            
            if video_id not in video_scores:
                video_scores[video_id] = {
                    'score': 0,
                    'items': []
                }
            
            video_scores[video_id]['score'] += score
            video_scores[video_id]['items'].append(result)
        
        # Adicionar resultados de imagem
        for i, result in enumerate(results_image):
            video_id = result['video_id']
            # Pontuação inversa à posição no ranking, normalizada
            score = (len(results_image) - i) / len(results_image) * weights[1]
            
            if video_id not in video_scores:
                video_scores[video_id] = {
                    'score': 0,
                    'items': []
                }
            
            video_scores[video_id]['score'] += score
            video_scores[video_id]['items'].append(result)
        
        # Ordenar vídeos por pontuação
        sorted_videos = sorted(video_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        
        # Criar lista final de resultados
        combined_results = []
        for video_id, data in sorted_videos[:limit]:
            # Ordenar itens do vídeo por pontuação
            items = sorted(data['items'], key=lambda x: x.get('score', 0), reverse=True)
            
            # Adicionar itens à lista final
            for item in items:
                if item not in combined_results:
                    combined_results.append(item)
        
        return combined_results[:limit]
