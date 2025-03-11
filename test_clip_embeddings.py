#!/usr/bin/env python3
"""
Script para testar se os embeddings de texto e imagem do CLIP estão no mesmo espaço vetorial.
Este script demonstra como os embeddings de texto e imagem podem ser comparados diretamente.
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from src.models.image_models import ImageEncoder
from src.models.text_models import TextEncoder

def test_clip_embeddings(image_path=None):
    """
    Testa a compatibilidade entre embeddings de texto e imagem do CLIP.
    
    Args:
        image_path: Caminho para uma imagem de teste. Se None, usa uma imagem de exemplo.
    """
    # Criar diretório para imagens de teste se não existir
    test_dir = "test_images"
    os.makedirs(test_dir, exist_ok=True)
    
    # Se nenhuma imagem for fornecida, usar uma imagem de exemplo
    if image_path is None or not os.path.exists(image_path):
        # Criar uma imagem simples para teste
        image_path = os.path.join(test_dir, "test_image.jpg")
        if not os.path.exists(image_path):
            print(f"Criando imagem de teste em {image_path}")
            # Criar uma imagem simples com um círculo
            img = Image.new('RGB', (300, 300), color='white')
            plt.figure(figsize=(3, 3))
            plt.imshow(img)
            circle = plt.Circle((150, 150), 100, fill=True, color='blue')
            plt.gca().add_patch(circle)
            plt.axis('off')
            plt.savefig(image_path)
            plt.close()
    
    # Inicializar os codificadores
    image_encoder = ImageEncoder()
    text_encoder = TextEncoder()
    
    # Verificar se estão usando o mesmo modelo
    print("\nVerificando modelos:")
    print(f"ImageEncoder: {image_encoder.model.config._name_or_path}")
    print(f"TextEncoder: {text_encoder.model.config._name_or_path}")
    
    # Carregar a imagem
    image = Image.open(image_path).convert("RGB")
    
    # Definir textos para teste
    texts = [
        "um círculo azul",
        "um círculo vermelho",
        "um quadrado azul",
        "uma pessoa caminhando",
        "um cachorro",
        "um gato",
        "uma paisagem",
        "um carro"
    ]
    
    # Codificar a imagem
    image_embedding = image_encoder.encode_image(image)
    
    # Codificar os textos
    text_embeddings = [text_encoder.encode_text(text) for text in texts]
    
    # Calcular similaridades
    similarities = [np.dot(image_embedding, text_embedding) for text_embedding in text_embeddings]
    
    # Exibir resultados
    print("\nSimilaridades entre a imagem e os textos:")
    print("-" * 50)
    
    # Ordenar por similaridade
    sorted_indices = np.argsort(similarities)[::-1]
    
    for i in sorted_indices:
        print(f"{texts[i]:<20}: {similarities[i]:.4f}")
    
    # Verificar se as dimensões dos embeddings são iguais
    print("\nDimensões dos embeddings:")
    print(f"Imagem: {image_embedding.shape}")
    print(f"Texto: {text_embeddings[0].shape}")
    
    # Verificar se os embeddings estão normalizados
    print("\nNormas dos embeddings:")
    print(f"Imagem: {np.linalg.norm(image_embedding):.4f}")
    print(f"Texto: {np.linalg.norm(text_embeddings[0]):.4f}")
    
    # Exibir a imagem
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.title("Imagem de teste")
    plt.axis('off')
    plt.show()
    
    return {
        'image_embedding': image_embedding,
        'text_embeddings': text_embeddings,
        'similarities': similarities,
        'texts': texts
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Testar embeddings de texto e imagem do CLIP")
    parser.add_argument("--image", help="Caminho para uma imagem de teste")
    
    args = parser.parse_args()
    
    test_clip_embeddings(args.image) 