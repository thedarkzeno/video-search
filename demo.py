import os
import gradio as gr
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

from src.search import VideoSearch

# Inicializar o motor de busca
search_engine = VideoSearch(output_dir="processed_data")

def display_results(results, num_results=5):
    """
    Formata os resultados da busca para exibição na interface Gradio.
    """
    if not results:
        return "Nenhum resultado encontrado."
    
    results = results[:num_results]
    
    # Criar listas para as imagens e seus metadados
    images = []
    captions = []
    
    for result in results:
        frame_path = result['frame_path']
        description = result['description']
        video_path = result['video_path']
        timestamp = result['timestamp']
        score = result['score']
        match_type = result.get('match_type', 'desconhecido')
        
        # Carregar a imagem
        try:
            img = Image.open(frame_path)
            images.append(img)
            
            # Criar legenda para a imagem
            caption = f"Vídeo: {os.path.basename(video_path)} | Timestamp: {timestamp:.2f}s | Score: {score:.4f} ({match_type})\n{description}"
            captions.append(caption)
        except Exception as e:
            print(f"Erro ao carregar imagem {frame_path}: {e}")
    
    # Se não foi possível carregar nenhuma imagem
    if not images:
        return "Não foi possível carregar as imagens dos resultados."
    
    # Criar uma galeria de imagens
    gallery = gr.Gallery(value=[(img, cap) for img, cap in zip(images, captions)])
    return gallery

def search_by_text(query_text, semantic_weight, max_results):
    """
    Realiza a busca por texto e retorna os resultados formatados.
    """
    if not query_text.strip():
        return "Por favor, insira um texto para busca."
    
    results = search_engine.search_by_text(
        query_text, 
        limit=max_results,
        semantic_weight=semantic_weight
    )
    
    return display_results(results, max_results)

def search_by_image(query_image, max_results):
    """
    Realiza a busca por imagem e retorna os resultados formatados.
    """
    if query_image is None:
        return "Por favor, carregue uma imagem para busca."
    
    # Salvar a imagem temporariamente
    temp_image_path = "temp_query_image.jpg"
    query_image.save(temp_image_path)
    
    # Realizar a busca
    results = search_engine.search_by_image(
        temp_image_path,
        limit=max_results
    )
    
    # Remover a imagem temporária
    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)
    
    return display_results(results, max_results)

def search_multimodal(query_text, query_image, text_weight, max_results):
    """
    Realiza a busca multimodal (texto + imagem) e retorna os resultados formatados.
    """
    if not query_text.strip() and query_image is None:
        return "Por favor, insira um texto ou carregue uma imagem para busca."
    
    # Salvar a imagem temporariamente se existir
    temp_image_path = None
    if query_image is not None:
        temp_image_path = "temp_query_image.jpg"
        query_image.save(temp_image_path)
    
    # Realizar a busca multimodal
    results = search_engine.search_multimodal(
        query_text=query_text if query_text.strip() else None,
        query_image=temp_image_path,
        weights=(text_weight, 1.0 - text_weight),
        limit=max_results
    )
    
    # Remover a imagem temporária
    if temp_image_path and os.path.exists(temp_image_path):
        os.remove(temp_image_path)
    
    return display_results(results, max_results)

def create_demo():
    # Verificar se o banco de dados existe
    db_path = search_engine.get_db_path()
    if not os.path.exists(db_path):
        print(f"Erro: Banco de dados não encontrado em {db_path}")
        print("Execute o processamento de vídeos antes de iniciar a demo.")
        return None
    
    # Interface de busca
    with gr.Blocks(title="Framework de Busca em Vídeos") as demo:
        gr.Markdown("# Framework de Busca em Vídeos")
        
        with gr.Tabs():
            with gr.TabItem("Busca por Texto"):
                with gr.Row():
                    with gr.Column(scale=3):
                        text_input = gr.Textbox(
                            label="Digite o texto para busca", 
                            placeholder="Ex: pessoa caminhando na praia"
                        )
                    with gr.Column(scale=1):
                        semantic_weight = gr.Slider(
                            minimum=0.0, 
                            maximum=1.0, 
                            value=0.5, 
                            step=0.1, 
                            label="Peso Semântico"
                        )
                        max_results_text = gr.Slider(
                            minimum=1, 
                            maximum=20, 
                            value=1, 
                            step=1, 
                            label="Número de Resultados"
                        )
                
                text_search_button = gr.Button("Buscar por Texto")
                text_results = gr.Gallery(label="Resultados")
                text_details = gr.HTML(label="Detalhes")
                
                # Modificar a função de busca para retornar tanto imagens quanto detalhes
                def text_search_with_gallery(query, weight, max_results):
                    results = search_engine.search_by_text(query, limit=max_results, semantic_weight=weight)
                    if not results:
                        return [], "Nenhum resultado encontrado."
                    
                    # Preparar galeria
                    gallery_items = []
                    details_html = "<div style='display: flex; flex-direction: column; gap: 20px;'>"
                    
                    for i, result in enumerate(results):
                        if i >= max_results:
                            break
                        
                        try:
                            # Carregar imagem para galeria
                            img = Image.open(result['frame_path'])
                            caption = f"Score: {result['score']:.4f} - {os.path.basename(result['video_path'])} ({result['timestamp']:.2f}s)"
                            gallery_items.append((img, caption))
                            
                            # Preparar detalhes
                            details_html += f"""
                            <div style='border: 1px solid #ddd; border-radius: 8px; padding: 15px;'>
                                <h3>Pontuação: {result['score']:.4f} ({result.get('match_type', 'desconhecido')})</h3>
                                <p><b>Vídeo:</b> {os.path.basename(result['video_path'])}</p>
                                <p><b>Timestamp:</b> {result['timestamp']:.2f}s</p>
                                <p><b>Descrição:</b> {result['description']}</p>
                            </div>
                            """
                        except Exception as e:
                            print(f"Erro ao processar resultado: {e}")
                    
                    details_html += "</div>"
                    return gallery_items, details_html
                
                text_search_button.click(
                    fn=text_search_with_gallery,
                    inputs=[text_input, semantic_weight, max_results_text],
                    outputs=[text_results, text_details]
                )
            
            with gr.TabItem("Busca por Imagem"):
                with gr.Row():
                    with gr.Column(scale=3):
                        image_input = gr.Image(type="pil", label="Carregue uma imagem para busca")
                    with gr.Column(scale=1):
                        max_results_image = gr.Slider(
                            minimum=1, 
                            maximum=20, 
                            value=5, 
                            step=1, 
                            label="Número de Resultados"
                        )
                
                image_search_button = gr.Button("Buscar por Imagem")
                image_results = gr.Gallery(label="Resultados")
                
                image_search_button.click(
                    fn=search_by_image,
                    inputs=[image_input, max_results_image],
                    outputs=image_results
                )
            
            with gr.TabItem("Busca Multimodal"):
                with gr.Row():
                    with gr.Column(scale=2):
                        multi_text_input = gr.Textbox(
                            label="Digite o texto para busca (opcional)", 
                            placeholder="Ex: pessoa caminhando na praia"
                        )
                        multi_image_input = gr.Image(type="pil", label="Carregue uma imagem para busca (opcional)")
                    with gr.Column(scale=1):
                        text_weight = gr.Slider(
                            minimum=0.0, 
                            maximum=1.0, 
                            value=0.5, 
                            step=0.1, 
                            label="Peso do Texto vs. Imagem"
                        )
                        max_results_multi = gr.Slider(
                            minimum=1, 
                            maximum=20, 
                            value=5, 
                            step=1, 
                            label="Número de Resultados"
                        )
                
                multi_search_button = gr.Button("Buscar (Texto + Imagem)")
                multi_results = gr.Gallery(label="Resultados")
                
                multi_search_button.click(
                    fn=search_multimodal,
                    inputs=[multi_text_input, multi_image_input, text_weight, max_results_multi],
                    outputs=multi_results
                )
        
        gr.Markdown("""
        ### Como usar
        
        - **Busca por Texto**: Digite um texto descritivo e ajuste o peso semântico para equilibrar entre correspondência exata e conceitual
        - **Busca por Imagem**: Carregue uma imagem para encontrar frames visualmente similares 
        - **Busca Multimodal**: Combine texto e imagem para resultados mais precisos
        
        O peso semântico controla quanto da busca deve ser baseada em significado versus correspondência exata de palavras.
        """)
    
    return demo

if __name__ == "__main__":
    # Verificar se gradio está instalado
    try:
        import gradio as gr
    except ImportError:
        print("Gradio não está instalado. Instalando...")
        import subprocess
        subprocess.check_call(["pip", "install", "gradio"])
        import gradio as gr
    
    # Criar e iniciar o demo
    demo = create_demo()
    if demo:
        # Use o diretório de frames como diretório estático
        frames_dir = os.path.join("processed_data", "frames")
        print(f"Servindo imagens a partir de: {os.path.abspath(frames_dir)}")
        demo.launch(share=False, allowed_paths=[frames_dir])
    else:
        print("Falha ao iniciar o demo. Verifique se os dados foram processados.")