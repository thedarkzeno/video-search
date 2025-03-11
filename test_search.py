#!/usr/bin/env python3
"""
Teste da funcionalidade de busca por texto com semântica.
"""

from src.search import VideoSearch

def main():
    # Inicializar o mecanismo de busca
    search_engine = VideoSearch(output_dir="processed_data")
    
    # Testar busca por texto com diferentes pesos semânticos
    query = "airplane"
    
    # Busca principalmente textual
    print(f"\n{'='*50}")
    print(f"Busca principalmente textual (10% semântica): '{query}'")
    print(f"{'='*50}")
    results_textual = search_engine.search_by_text(query, semantic_weight=0.1)
    display_results(results_textual)
    
    # Busca equilibrada
    print(f"\n{'='*50}")
    print(f"Busca equilibrada (50% semântica): '{query}'")
    print(f"{'='*50}")
    results_balanced = search_engine.search_by_text(query, semantic_weight=0.5)
    display_results(results_balanced)
    
    # Busca principalmente semântica
    print(f"\n{'='*50}")
    print(f"Busca principalmente semântica (90% semântica): '{query}'")
    print(f"{'='*50}")
    results_semantic = search_engine.search_by_text(query, semantic_weight=0.9)
    display_results(results_semantic)
    
    # Testar com uma consulta diferente que pode se beneficiar da semântica
    semantic_query = "flying in the sky"
    print(f"\n{'='*50}")
    print(f"Busca semântica para: '{semantic_query}'")
    print(f"{'='*50}")
    results_semantic_query = search_engine.search_by_text(semantic_query, semantic_weight=0.8)
    display_results(results_semantic_query)

def display_results(results):
    # Exibir resultados
    if not results:
        print("Nenhum resultado encontrado.")
    else:
        print(f"\nResultados encontrados: {len(results)}")
        for i, result in enumerate(results):
            print(f"\n--- Resultado {i+1} ---")
            print(f"Vídeo: {result['video_path']}")
            
            if result['type'] == 'frame':
                print(f"Tipo: Frame")
                print(f"Timestamp: {result['timestamp']:.2f}s")
                print(f"Descrição: {result['description']}")
                print(f"Caminho do frame: {result['frame_path']}")
                if 'match_type' in result:
                    print(f"Tipo de correspondência: {result['match_type']}")
            else:
                print(f"Tipo: Transcrição de áudio")
                print(f"Intervalo: {result['start_time']:.2f}s - {result['end_time']:.2f}s")
                print(f"Texto: {result['text']}")
                if 'match_type' in result:
                    print(f"Tipo de correspondência: {result['match_type']}")
            
            print(f"Pontuação: {result.get('score', 0):.4f}")

if __name__ == "__main__":
    main() 