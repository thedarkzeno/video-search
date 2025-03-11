#!/usr/bin/env python3
"""
Script para verificar o status do banco de dados.
Mostra informações sobre os vídeos, frames e transcrições armazenados.
"""

import os
import argparse
import sqlite3
import faiss
import pickle
import humanize
from tabulate import tabulate
from datetime import datetime

def get_database_status(output_dir="processed_data"):
    """
    Obtém o status do banco de dados.
    
    Args:
        output_dir: Diretório onde estão os dados processados.
    
    Returns:
        dict: Informações sobre o banco de dados.
    """
    # Verificar se o diretório existe
    if not os.path.exists(output_dir):
        print(f"Diretório '{output_dir}' não encontrado.")
        return None
    
    # Caminhos para o banco de dados e índice
    db_path = os.path.join(output_dir, "video_search.db")
    index_path = os.path.join(output_dir, "embeddings_index")
    index_file = os.path.join(index_path, "frame_embeddings.index")
    id_map_file = os.path.join(index_path, "id_mapping.pkl")
    frames_dir = os.path.join(output_dir, "frames")
    audio_dir = os.path.join(output_dir, "audio")
    
    # Verificar se o banco de dados existe
    if not os.path.exists(db_path):
        print(f"Banco de dados não encontrado: {db_path}")
        return None
    
    # Conectar ao banco de dados
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Obter informações sobre os vídeos
    cursor.execute("SELECT COUNT(*) FROM videos")
    video_count = cursor.fetchone()[0]
    
    # Obter informações sobre os frames
    cursor.execute("SELECT COUNT(*) FROM key_frames")
    frame_count = cursor.fetchone()[0]
    
    # Obter informações sobre as transcrições
    cursor.execute("SELECT COUNT(*) FROM audio_transcriptions")
    transcription_count = cursor.fetchone()[0]
    
    # Obter informações sobre os vídeos processados
    videos = []
    cursor.execute("""
        SELECT v.id, v.file_path, v.duration, v.processed_at,
               COUNT(DISTINCT k.id) as frame_count,
               COUNT(DISTINCT a.id) as transcription_count
        FROM videos v
        LEFT JOIN key_frames k ON v.id = k.video_id
        LEFT JOIN audio_transcriptions a ON v.id = a.video_id
        GROUP BY v.id
        ORDER BY v.processed_at DESC
    """)
    
    for row in cursor.fetchall():
        video_id, file_path, duration, processed_at, video_frame_count, video_transcription_count = row
        videos.append({
            'id': video_id,
            'file_path': os.path.basename(file_path),
            'duration': duration,
            'processed_at': processed_at,
            'frame_count': video_frame_count,
            'transcription_count': video_transcription_count
        })
    
    conn.close()
    
    # Verificar o índice FAISS
    index_size = 0
    index_count = 0
    if os.path.exists(index_file):
        index_size = os.path.getsize(index_file)
        try:
            index = faiss.read_index(index_file)
            index_count = index.ntotal
        except:
            index_count = "Erro ao ler índice"
    
    # Verificar o mapeamento de IDs
    id_map_count = 0
    if os.path.exists(id_map_file):
        try:
            with open(id_map_file, 'rb') as f:
                id_mapping = pickle.load(f)
                id_map_count = len(id_mapping)
        except:
            id_map_count = "Erro ao ler mapeamento"
    
    # Verificar os diretórios de frames e áudio
    frames_count = 0
    frames_size = 0
    if os.path.exists(frames_dir):
        for root, dirs, files in os.walk(frames_dir):
            frames_count += len(files)
            frames_size += sum(os.path.getsize(os.path.join(root, file)) for file in files)
    
    audio_count = 0
    audio_size = 0
    if os.path.exists(audio_dir):
        for root, dirs, files in os.walk(audio_dir):
            audio_count += len(files)
            audio_size += sum(os.path.getsize(os.path.join(root, file)) for file in files)
    
    # Calcular o tamanho total do banco de dados
    db_size = os.path.getsize(db_path) if os.path.exists(db_path) else 0
    
    # Calcular o tamanho total dos dados
    total_size = db_size + index_size + frames_size + audio_size
    
    return {
        'db_path': db_path,
        'index_path': index_path,
        'video_count': video_count,
        'frame_count': frame_count,
        'transcription_count': transcription_count,
        'index_count': index_count,
        'id_map_count': id_map_count,
        'frames_count': frames_count,
        'frames_size': frames_size,
        'audio_count': audio_count,
        'audio_size': audio_size,
        'db_size': db_size,
        'index_size': index_size,
        'total_size': total_size,
        'videos': videos
    }

def display_status(status):
    """
    Exibe o status do banco de dados de forma formatada.
    
    Args:
        status: Informações sobre o banco de dados.
    """
    if status is None:
        return
    
    print("\n" + "="*80)
    print(f"STATUS DO BANCO DE DADOS")
    print("="*80)
    
    print(f"\nLocalização:")
    print(f"  Banco de dados: {status['db_path']}")
    print(f"  Índice FAISS: {status['index_path']}")
    
    print(f"\nEstatísticas gerais:")
    print(f"  Vídeos processados: {status['video_count']}")
    print(f"  Frames principais: {status['frame_count']}")
    print(f"  Transcrições de áudio: {status['transcription_count']}")
    print(f"  Embeddings no índice FAISS: {status['index_count']}")
    
    print(f"\nArmazenamento:")
    print(f"  Tamanho do banco de dados: {humanize.naturalsize(status['db_size'])}")
    print(f"  Tamanho do índice FAISS: {humanize.naturalsize(status['index_size'])}")
    print(f"  Tamanho dos frames: {humanize.naturalsize(status['frames_size'])} ({status['frames_count']} arquivos)")
    print(f"  Tamanho dos áudios: {humanize.naturalsize(status['audio_size'])} ({status['audio_count']} arquivos)")
    print(f"  Tamanho total: {humanize.naturalsize(status['total_size'])}")
    
    if status['videos']:
        print(f"\nVídeos processados:")
        
        # Preparar dados para a tabela
        table_data = []
        for video in status['videos']:
            # Formatar a data de processamento
            try:
                processed_at = datetime.fromisoformat(video['processed_at']).strftime('%Y-%m-%d %H:%M:%S')
            except:
                processed_at = video['processed_at']
            
            # Formatar a duração
            duration = f"{video['duration']:.2f}s ({int(video['duration'] // 60)}m {int(video['duration'] % 60)}s)"
            
            table_data.append([
                video['id'],
                video['file_path'],
                duration,
                processed_at,
                video['frame_count'],
                video['transcription_count']
            ])
        
        # Exibir a tabela
        headers = ["ID", "Arquivo", "Duração", "Processado em", "Frames", "Transcrições"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    print("\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(description="Verificar o status do banco de dados")
    parser.add_argument("--output-dir", default="processed_data", help="Diretório onde estão os dados processados")
    
    args = parser.parse_args()
    
    status = get_database_status(args.output_dir)
    display_status(status)

if __name__ == "__main__":
    main() 