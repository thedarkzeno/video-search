#!/usr/bin/env python3
"""
Exemplo de uso do framework de processamento e busca de vídeos.
"""

import os
import argparse
from pathlib import Path

from src.processor import VideoProcessor
from src.search import VideoSearch

def process_videos(video_paths, output_dir="processed_data"):
    """
    Processa uma lista de vídeos.
    
    Args:
        video_paths: Lista de caminhos para vídeos.
        output_dir: Diretório para salvar os dados processados.
    """
    # Criar diretório de saída
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Inicializar o processador
    processor = VideoProcessor(
        output_dir=output_dir,
        frame_similarity_threshold=0.8,  # Ajuste conforme necessário
        frame_interval=1.0               # Intervalo em segundos
    )
    
    # Processar cada vídeo
    for video_path in video_paths:
        try:
            print(f"\n{'='*50}")
            print(f"Processando vídeo: {video_path}")
            print(f"{'='*50}")
            
            video_id = processor.process_video(video_path)
            print(f"Vídeo processado com sucesso! ID: {video_id}")
            
        except Exception as e:
            print(f"Erro ao processar o vídeo {video_path}: {str(e)}")

def search_videos(query_text=None, query_image=None, output_dir="processed_data"):
    """
    Realiza uma busca nos vídeos processados.
    
    Args:
        query_text: Texto para busca.
        query_image: Caminho para imagem para busca.
        output_dir: Diretório onde estão os dados processados.
    """
    # Inicializar o mecanismo de busca
    search_engine = VideoSearch(output_dir=output_dir)
    
    # Realizar busca
    if query_text and query_image:
        print(f"\n{'='*50}")
        print(f"Busca multimodal - Texto: '{query_text}', Imagem: '{query_image}'")
        print(f"{'='*50}")
        
        results = search_engine.search_multimodal(
            query_text=query_text,
            query_image=query_image,
            weights=(0.5, 0.5),
            limit=10
        )
    elif query_text:
        print(f"\n{'='*50}")
        print(f"Busca por texto: '{query_text}'")
        print(f"{'='*50}")
        
        results = search_engine.search_by_text(query_text, limit=10)
    elif query_image:
        print(f"\n{'='*50}")
        print(f"Busca por imagem: '{query_image}'")
        print(f"{'='*50}")
        
        results = search_engine.search_by_image(query_image, limit=10)
    else:
        print("Erro: Forneça um texto ou uma imagem para busca.")
        return
    
    # Exibir resultados
    if not results:
        print("Nenhum resultado encontrado.")
        return
    
    print(f"\nResultados encontrados: {len(results)}")
    for i, result in enumerate(results):
        print(f"\n--- Resultado {i+1} ---")
        print(f"Vídeo: {result['video_path']}")
        
        if result['type'] == 'frame':
            print(f"Tipo: Frame")
            print(f"Timestamp: {result['timestamp']:.2f}s")
            print(f"Descrição: {result['description']}")
            print(f"Caminho do frame: {result['frame_path']}")
        else:
            print(f"Tipo: Transcrição de áudio")
            print(f"Intervalo: {result['start_time']:.2f}s - {result['end_time']:.2f}s")
            print(f"Texto: {result['text']}")
        
        print(f"Pontuação: {result.get('score', 0):.4f}")

def main():
    parser = argparse.ArgumentParser(description="Framework de processamento e busca de vídeos")
    subparsers = parser.add_subparsers(dest="command", help="Comando a ser executado")
    
    # Subparser para processamento
    process_parser = subparsers.add_parser("process", help="Processar vídeos")
    process_parser.add_argument("video_paths", nargs="+", help="Caminhos para os vídeos a serem processados")
    process_parser.add_argument("--output-dir", default="processed_data", help="Diretório para salvar os dados processados")
    
    # Subparser para busca
    search_parser = subparsers.add_parser("search", help="Buscar em vídeos processados")
    search_parser.add_argument("--text", help="Texto para busca")
    search_parser.add_argument("--image", help="Caminho para imagem para busca")
    search_parser.add_argument("--output-dir", default="processed_data", help="Diretório onde estão os dados processados")
    
    args = parser.parse_args()
    
    if args.command == "process":
        process_videos(args.video_paths, args.output_dir)
    elif args.command == "search":
        search_videos(args.text, args.image, args.output_dir)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 