#!/usr/bin/env python3
"""
Script para resetar o banco de dados e o índice FAISS.
Este script remove todos os dados processados e permite começar do zero.
"""

import os
import shutil
import argparse
from pathlib import Path

def reset_database(output_dir="processed_data", confirm=False):
    """
    Reseta o banco de dados e o índice FAISS.
    
    Args:
        output_dir: Diretório onde estão os dados processados.
        confirm: Se True, não pede confirmação antes de resetar.
    
    Returns:
        bool: True se o reset foi bem-sucedido, False caso contrário.
    """
    # Verificar se o diretório existe
    if not os.path.exists(output_dir):
        print(f"Diretório '{output_dir}' não encontrado. Nada para resetar.")
        return False
    
    # Caminhos para o banco de dados e índice
    db_path = os.path.join(output_dir, "video_search.db")
    index_path = os.path.join(output_dir, "embeddings_index")
    frames_dir = os.path.join(output_dir, "frames")
    audio_dir = os.path.join(output_dir, "audio")
    
    # Listar arquivos e diretórios a serem removidos
    items_to_remove = []
    
    if os.path.exists(db_path):
        items_to_remove.append(db_path)
    
    if os.path.exists(index_path):
        items_to_remove.append(index_path)
    
    if os.path.exists(frames_dir):
        items_to_remove.append(frames_dir)
    
    if os.path.exists(audio_dir):
        items_to_remove.append(audio_dir)
    
    if not items_to_remove:
        print(f"Nenhum dado encontrado em '{output_dir}'. Nada para resetar.")
        return False
    
    # Mostrar o que será removido
    print("Os seguintes itens serão removidos:")
    for item in items_to_remove:
        print(f"  - {item}")
    
    # Pedir confirmação se necessário
    if not confirm:
        response = input("\nTem certeza que deseja resetar o banco de dados? Esta ação não pode ser desfeita. [s/N]: ")
        if response.lower() not in ["s", "sim", "y", "yes"]:
            print("Operação cancelada.")
            return False
    
    # Remover os itens
    for item in items_to_remove:
        try:
            if os.path.isfile(item):
                os.remove(item)
                print(f"Arquivo removido: {item}")
            elif os.path.isdir(item):
                shutil.rmtree(item)
                print(f"Diretório removido: {item}")
        except Exception as e:
            print(f"Erro ao remover {item}: {str(e)}")
    
    print("\nReset concluído com sucesso!")
    print(f"O banco de dados e os índices em '{output_dir}' foram removidos.")
    print("Você pode processar novos vídeos agora.")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Resetar o banco de dados e o índice FAISS")
    parser.add_argument("--output-dir", default="processed_data", help="Diretório onde estão os dados processados")
    parser.add_argument("--yes", "-y", action="store_true", help="Não pedir confirmação antes de resetar")
    
    args = parser.parse_args()
    
    reset_database(args.output_dir, args.yes)

if __name__ == "__main__":
    main() 