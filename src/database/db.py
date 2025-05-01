import sqlite3
import os
import json
import numpy as np
import faiss
import pickle
from pathlib import Path
import time

class DBManager:
    def __init__(self, db_path="video_search.db", index_path="embeddings_index"):
        """
        Inicializa o gerenciador de banco de dados.
        
        Args:
            db_path (str): Caminho para o arquivo SQLite
            index_path (str): Diretório para armazenar os índices FAISS
        """
        self.db_path = db_path
        self.index_path = index_path
        self.index_file = os.path.join(index_path, "frame_embeddings.index")
        self.id_map_file = os.path.join(index_path, "id_mapping.pkl")
        
        # Garantir que o diretório de índice existe
        Path(index_path).mkdir(parents=True, exist_ok=True)
        
        # Inicializar banco de dados
        self._init_db()
        
        # Inicializar índice FAISS
        self._init_faiss()
    
    def _init_db(self):
        """Inicializa o esquema do banco de dados SQLite."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Tabela de vídeos
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE,
            duration REAL,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processing_time_seconds REAL, 
            frame_processing_time_seconds REAL, 
            audio_processing_time_seconds REAL  
        )
        ''')
        
        # Add columns if they don't exist (for existing databases)
        columns_to_add = {
            'processing_time_seconds': 'REAL',
            'frame_processing_time_seconds': 'REAL',
            'audio_processing_time_seconds': 'REAL'
        }
        for col_name, col_type in columns_to_add.items():
            try:
                cursor.execute(f"ALTER TABLE videos ADD COLUMN {col_name} {col_type}")
                print(f"Added '{col_name}' column to 'videos' table.")
            except sqlite3.OperationalError as e:
                # Ignore error if column already exists
                if "duplicate column name" not in str(e).lower():
                    print(f"Could not add column {col_name}: {e}") # Print other errors
                    # Depending on the error, you might want to raise it
                    # raise e
        
        # Tabela de frames principais
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS key_frames (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER,
            timestamp REAL,
            frame_path TEXT,
            description TEXT,
            FOREIGN KEY (video_id) REFERENCES videos (id)
        )
        ''')
        
        # Tabela de transcrições de áudio
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS audio_transcriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER,
            start_time REAL,
            end_time REAL,
            text TEXT,
            FOREIGN KEY (video_id) REFERENCES videos (id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def _init_faiss(self):
        """Inicializa o índice FAISS para busca por similaridade."""
        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.id_map_file, 'rb') as f:
                self.id_mapping = pickle.load(f)
        else:
            # Dimensão do embedding (ajuste conforme o modelo usado)
            embedding_dim = 1152
            self.index = faiss.IndexFlatL2(embedding_dim)
            self.id_mapping = {}
    
    def _get_connection(self):
        """
        Obtém uma conexão com o banco de dados SQLite.
        
        Returns:
            sqlite3.Connection: Conexão com o banco de dados.
        """
        conn = sqlite3.connect(self.db_path)
        return conn
    
    def add_video(self, file_path, duration):
        """
        Adiciona um novo vídeo ao banco de dados.
        
        Args:
            file_path (str): Caminho do arquivo de vídeo
            duration (float): Duração do vídeo em segundos
            
        Returns:
            int: ID do vídeo adicionado
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT OR IGNORE INTO videos (file_path, duration) VALUES (?, ?)",
            (file_path, duration)
        )
        
        # Obter o ID do vídeo (seja recém-inserido ou existente)
        cursor.execute("SELECT id FROM videos WHERE file_path = ?", (file_path,))
        video_id = cursor.fetchone()[0]
        
        conn.commit()
        conn.close()
        
        return video_id
    
    def add_key_frame(self, video_id, timestamp, frame_path, description):
        """
        Adiciona um frame principal ao banco de dados.
        
        Args:
            video_id (int): ID do vídeo
            timestamp (float): Timestamp do frame em segundos
            frame_path (str): Caminho para o arquivo de imagem do frame
            description (str): Descrição gerada para o frame
            
        Returns:
            int: ID do frame adicionado
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO key_frames (video_id, timestamp, frame_path, description) VALUES (?, ?, ?, ?)",
            (video_id, timestamp, frame_path, description)
        )
        
        frame_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        return frame_id
    
    def add_audio_transcription(self, video_id, start_time, end_time, text):
        """
        Adiciona uma transcrição de áudio ao banco de dados.
        
        Args:
            video_id (int): ID do vídeo
            start_time (float): Tempo de início do segmento em segundos
            end_time (float): Tempo de fim do segmento em segundos
            text (str): Texto transcrito
            
        Returns:
            int: ID da transcrição adicionada
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO audio_transcriptions (video_id, start_time, end_time, text) VALUES (?, ?, ?, ?)",
            (video_id, start_time, end_time, text)
        )
        
        conn.commit()
        conn.close()
    
    def add_frame_embedding(self, frame_id, embedding):
        """
        Adiciona um embedding de frame ao índice FAISS.
        
        Args:
            frame_id (int): ID do frame no banco de dados
            embedding (numpy.ndarray): Vetor de embedding do frame
        """
        # Converter para o formato correto para FAISS
        embedding = np.array([embedding], dtype=np.float32)
        
        # Adicionar ao índice
        index_id = self.index.ntotal
        self.index.add(embedding)
        
        # Mapear o ID do índice para o ID do frame
        self.id_mapping[index_id] = frame_id
        
        # Salvar o índice atualizado
        faiss.write_index(self.index, self.index_file)
        with open(self.id_map_file, 'wb') as f:
            pickle.dump(self.id_mapping, f)
    
    def search_by_embedding(self, query_embedding, top_k=5):
        """
        Busca frames similares usando um embedding de consulta.
        
        Args:
            query_embedding (numpy.ndarray): Vetor de embedding da consulta
            top_k (int): Número de resultados a retornar
            
        Returns:
            list: Lista de IDs de frames e suas pontuações
        """
        query_embedding = np.array([query_embedding], dtype=np.float32)
        
        # Realizar a busca
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Mapear índices FAISS para IDs de frames
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1 and idx in self.id_mapping:
                frame_id = self.id_mapping[idx]
                results.append((frame_id, float(dist)))
        
        return results
    
    def search_by_text(self, query_text, embedding=None, semantic_weight=0.5, top_k_semantic=20):
        """
        Busca por texto nas descrições de frames e transcrições de áudio.
        Combina busca textual (correspondências parciais) com busca semântica (embeddings).
        
        Args:
            query_text (str): Texto para buscar
            embedding (numpy.ndarray, optional): Embedding do texto de consulta. Se None, apenas busca textual é realizada.
            semantic_weight (float): Peso para a busca semântica (0-1). Padrão: 0.5
            top_k_semantic (int): Número de resultados a retornar na busca semântica. Padrão: 20
            
        Returns:
            list: Resultados ordenados por relevância combinada
        """
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Resultados para frames e transcrições
        frame_results = []
        audio_results = []
        
        # Resultados da busca semântica (por embedding)
        semantic_results = []
        
        # Tokenizar a consulta (dividir em palavras)
        tokens = query_text.lower().split()
        
        # Realizar busca textual
        if tokens:
            # Buscar em descrições de frames
            frame_conditions = []
            frame_params = []
            
            for token in tokens:
                # Adicionar condição para cada token
                frame_conditions.append("LOWER(k.description) LIKE ?")
                frame_params.append(f'%{token}%')
            
            # Construir a consulta SQL
            frame_query = f"""
                SELECT v.id as video_id, v.file_path, k.id as frame_id, 
                       k.timestamp, k.frame_path, k.description
                FROM key_frames k
                JOIN videos v ON k.video_id = v.id
                WHERE {" OR ".join(frame_conditions)}
            """
            
            cursor.execute(frame_query, frame_params)
            frame_rows = cursor.fetchall()
            
            # Calcular pontuação para cada resultado
            for row in frame_rows:
                row_dict = dict(row)
                description = row_dict['description'].lower()
                
                # Calcular pontuação baseada no número de tokens encontrados
                score = sum(1 for token in tokens if token in description)
                # Normalizar pontuação (0-1)
                score = score / len(tokens)
                
                # Ajustar pontuação pelo peso da busca textual
                score = score * (1 - semantic_weight)
                
                row_dict['score'] = score
                row_dict['match_type'] = 'text'
                frame_results.append(row_dict)
            
            # Buscar em transcrições de áudio
            audio_conditions = []
            audio_params = []
            
            for token in tokens:
                # Adicionar condição para cada token
                audio_conditions.append("LOWER(a.text) LIKE ?")
                audio_params.append(f'%{token}%')
            
            # Construir a consulta SQL
            audio_query = f"""
                SELECT v.id as video_id, v.file_path, a.id as transcription_id,
                       a.start_time, a.end_time, a.text
                FROM audio_transcriptions a
                JOIN videos v ON a.video_id = v.id
                WHERE {" OR ".join(audio_conditions)}
            """
            
            cursor.execute(audio_query, audio_params)
            audio_rows = cursor.fetchall()
            
            # Calcular pontuação para cada resultado
            for row in audio_rows:
                row_dict = dict(row)
                text = row_dict['text'].lower()
                
                # Calcular pontuação baseada no número de tokens encontrados
                score = sum(1 for token in tokens if token in text)
                # Normalizar pontuação (0-1)
                score = score / len(tokens)
                
                # Ajustar pontuação pelo peso da busca textual
                score = score * (1 - semantic_weight)
                
                row_dict['score'] = score
                row_dict['match_type'] = 'text'
                audio_results.append(row_dict)
        
        # Realizar busca semântica se o embedding for fornecido
        if embedding is not None:
            # Buscar frames similares usando o embedding
            semantic_matches = self.search_by_embedding(embedding, top_k=top_k_semantic)
            
            # Obter informações dos frames encontrados
            for frame_id, distance in semantic_matches:
                # Converter distância em pontuação (menor distância = maior pontuação)
                # Normalizar para o intervalo [0, 1]
                similarity_score = 1.0 / (1.0 + distance)
                
                # Ajustar pontuação pelo peso da busca semântica
                score = similarity_score * semantic_weight
                
                # Obter informações do frame
                cursor.execute("""
                    SELECT v.id as video_id, v.file_path, k.id as frame_id, 
                           k.timestamp, k.frame_path, k.description
                    FROM key_frames k
                    JOIN videos v ON k.video_id = v.id
                    WHERE k.id = ?
                """, (frame_id,))
                
                row = cursor.fetchone()
                if row:
                    row_dict = dict(row)
                    row_dict['score'] = score
                    row_dict['match_type'] = 'semantic'
                    semantic_results.append(row_dict)
        
        conn.close()
        
        # Combinar resultados de busca textual e semântica
        all_results = frame_results + audio_results + semantic_results
        
        # Agrupar resultados por ID (para combinar pontuações de diferentes tipos de busca)
        grouped_results = {}
        for result in all_results:
            result_id = None
            if 'frame_id' in result:
                result_id = f"frame_{result['frame_id']}"
            elif 'transcription_id' in result:
                result_id = f"transcription_{result['transcription_id']}"
            
            if result_id:
                if result_id not in grouped_results:
                    grouped_results[result_id] = result.copy()
                else:
                    # Se o resultado já existe, somar as pontuações
                    grouped_results[result_id]['score'] += result['score']
                    # Marcar como resultado combinado
                    grouped_results[result_id]['match_type'] = 'combined'
        # print(grouped_results)
        # Converter de volta para lista e ordenar por pontuação
        combined_results = list(grouped_results.values())
        combined_results.sort(key=lambda x: x['score'], reverse=True)
        
        return combined_results

    def delete_key_frames_for_video(self, video_id):
        """Deleta todos os key frames associados a um video_id."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            # First, get the IDs of frames to be deleted (optional, needed for FAISS cleanup)
            # cursor.execute("SELECT id FROM key_frames WHERE video_id = ?", (video_id,))
            # frame_ids_to_delete = [row[0] for row in cursor.fetchall()]
            
            # Delete from DB
            cursor.execute("DELETE FROM key_frames WHERE video_id = ?", (video_id,))
            conn.commit()
            deleted_count = cursor.rowcount
            print(f"Deleted {deleted_count} key frames for video ID {video_id}.")
            
            # TODO: Implement FAISS index removal if needed
            # This requires finding the FAISS index IDs corresponding to frame_ids_to_delete
            # and using self.index.remove_ids(...) which requires IndexIDMap or careful handling.
            
        except sqlite3.Error as e:
            print(f"Database error deleting key frames for video ID {video_id}: {e}")
            conn.rollback()
        finally:
            conn.close()

    def delete_audio_transcriptions_for_video(self, video_id):
        """Deleta todas as transcrições de áudio associadas a um video_id."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM audio_transcriptions WHERE video_id = ?", (video_id,))
            conn.commit()
            deleted_count = cursor.rowcount
            print(f"Deleted {deleted_count} audio transcriptions for video ID {video_id}.")
        except sqlite3.Error as e:
            print(f"Database error deleting audio transcriptions for video ID {video_id}: {e}")
            conn.rollback()
        finally:
            conn.close()

    def update_video_processing_time(self, video_id, total_time, frame_time=None, audio_time=None):
        """Atualiza os tempos de processamento (total, frames, áudio) de um vídeo específico."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                UPDATE videos 
                SET processing_time_seconds = ?,
                    frame_processing_time_seconds = ?,
                    audio_processing_time_seconds = ?,
                    processed_at = CURRENT_TIMESTAMP 
                WHERE id = ?
            ''', (total_time, frame_time, audio_time, video_id))
            conn.commit()
            print(f"Updated processing times for video ID {video_id} (Total: {total_time:.2f}s, Frames: {frame_time:.2f}s, Audio: {audio_time:.2f}s).")
        except sqlite3.Error as e:
            print(f"Database error updating processing times for video ID {video_id}: {e}")
            conn.rollback() # Rollback on error
        finally:
            conn.close()