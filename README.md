# Framework de Processamento e Busca de Vídeos

Este framework permite processar vídeos para extrair frames principais, gerar descrições e transcrever áudio, além de realizar buscas por texto ou imagem nos vídeos processados.

## Funcionalidades

1. **Processamento de Vídeos**:
   - Extração de frames principais baseada em similaridade visual
   - Geração de descrições para os frames principais
   - Extração e transcrição de áudio
   - Armazenamento em banco de dados SQLite

2. **Busca em Vídeos**:
   - Busca por texto nas descrições de frames e transcrições de áudio
     - Suporta correspondências parciais e ranking por relevância
     - Encontra resultados mesmo quando apenas algumas palavras da consulta estão presentes
     - Combina busca textual com busca semântica por embeddings
     - Usa o mesmo modelo CLIP para embeddings de texto e imagem, garantindo compatibilidade
     - Ajuste flexível do peso entre busca textual e semântica
   - Busca por similaridade de imagem usando FAISS
   - Busca multimodal combinando texto e imagem

## Requisitos

- Python 3.8+
- OpenCV
- PyTorch
- Transformers (Hugging Face)
- FAISS
- Whisper
- Sentence Transformers
- SQLite3
- FFmpeg (para extração de áudio)

## Instalação

```bash
# Clonar o repositório
git clone https://github.com/seu-usuario/video-search.git
cd video-search

# Instalar dependências
pip install -r requirements.txt

# Instalar FFmpeg (se ainda não estiver instalado)
# No Ubuntu/Debian:
# sudo apt-get install ffmpeg
# No Windows, baixe de https://ffmpeg.org/download.html
```

## Uso

### Processamento de Vídeos

```bash
python example.py process caminho/para/video1.mp4 caminho/para/video2.mp4 --output-dir processed_data
```

### Resetar o Banco de Dados

Se você precisar limpar todos os dados processados e começar do zero:

```bash
python reset_database.py --output-dir processed_data
```

Ou para resetar sem confirmação:

```bash
python reset_database.py --output-dir processed_data --yes
```

Este script remove o banco de dados SQLite, o índice FAISS e os diretórios de frames e áudio.

### Verificar o Status do Banco de Dados

Para verificar o status do banco de dados e obter estatísticas sobre os vídeos processados:

```bash
python database_status.py --output-dir processed_data
```

Este script mostra informações detalhadas sobre:
- Quantidade de vídeos, frames e transcrições
- Tamanho do banco de dados, índice FAISS, frames e áudios
- Lista de vídeos processados com detalhes

### Testar Embeddings CLIP

Para verificar se os embeddings de texto e imagem estão no mesmo espaço vetorial:

```bash
python test_clip_embeddings.py --image caminho/para/imagem.jpg
```

Este script demonstra como os embeddings de texto e imagem do CLIP podem ser comparados diretamente, mostrando a similaridade entre uma imagem e vários textos.

### Busca por Texto

```bash
python example.py search --text "pessoa caminhando na praia" --output-dir processed_data
```

Você também pode usar o script de teste para verificar a funcionalidade de busca por texto com diferentes pesos semânticos:

```bash
python test_search.py
```

Este script demonstra como a busca semântica pode encontrar resultados relevantes mesmo quando as palavras exatas não estão presentes nas descrições ou transcrições.

### Busca por Imagem

```bash
python example.py search --image caminho/para/imagem.jpg --output-dir processed_data
```

### Busca Multimodal (Texto + Imagem)

```bash
python example.py search --text "pessoa caminhando na praia" --image caminho/para/imagem.jpg --output-dir processed_data
```

### Uso Programático

```python
from src.processor import VideoProcessor
from src.search import VideoSearch

# Processar um vídeo
processor = VideoProcessor(
    output_dir="processed_data",
    frame_similarity_threshold=0.8,
    frame_interval=1.0
)
video_id = processor.process_video("caminho/para/video.mp4")

# Buscar por texto (combinando busca textual e semântica)
search_engine = VideoSearch(output_dir="processed_data")
results = search_engine.search_by_text(
    "pessoa caminhando na praia", 
    semantic_weight=0.5  # Equilibrado entre textual e semântico
)

# Busca mais semântica (encontra conceitos relacionados)
results = search_engine.search_by_text(
    "pessoa caminhando na praia", 
    semantic_weight=0.8  # Prioriza semântica sobre correspondência exata
)

# Buscar por imagem
results = search_engine.search_by_image("caminho/para/imagem.jpg")

# Busca multimodal
results = search_engine.search_multimodal(
    query_text="pessoa caminhando na praia",
    query_image="caminho/para/imagem.jpg"
)
```

## Estrutura do Projeto

```
video-search/
├── src/
│   ├── __init__.py
│   ├── processor.py         # Processamento de vídeos
│   ├── search.py            # Busca em vídeos
│   ├── database/
│   │   ├── __init__.py
│   │   └── db.py            # Gerenciamento do banco de dados
│   ├── models/
│   │   ├── __init__.py
│   │   ├── image_models.py  # Modelos para processamento de imagens
│   │   └── text_models.py   # Modelos para processamento de texto/áudio
│   └── utils/               # Utilitários diversos
├── example.py               # Exemplo de uso
├── test_search.py           # Teste de busca por texto
├── test_clip_embeddings.py  # Teste de embeddings CLIP
├── reset_database.py        # Script para resetar o banco de dados
├── database_status.py       # Script para verificar o status do banco de dados
└── requirements.txt         # Dependências do projeto
```

## Como Funciona

1. **Extração de Frames Principais**:
   - O primeiro frame é sempre considerado principal
   - A cada intervalo de tempo, um novo frame é comparado com o último frame principal
   - Se a similaridade estiver abaixo do limiar, o novo frame é considerado principal

2. **Geração de Descrições**:
   - Utiliza um modelo de captioning para gerar descrições dos frames principais

3. **Transcrição de Áudio**:
   - Extrai o áudio do vídeo usando FFmpeg
   - Utiliza o modelo Whisper para transcrever o áudio em segmentos

4. **Armazenamento**:
   - Frames, descrições e transcrições são armazenados em um banco de dados SQLite
   - Embeddings são armazenados em um índice FAISS para busca eficiente

5. **Busca**:
   - Busca por texto: utiliza correspondência de texto nas descrições e transcrições
     - Tokeniza a consulta e busca por correspondências parciais
     - Calcula pontuação de relevância com base na quantidade de tokens encontrados
     - Combina com busca semântica usando embeddings de texto do CLIP
     - Usa o mesmo espaço vetorial para texto e imagem
     - Permite ajustar o peso entre busca textual e semântica
     - Ordena resultados por relevância combinada
   - Busca por imagem: codifica a imagem de consulta e busca por similaridade no índice FAISS
   - Busca multimodal: combina os resultados de busca por texto e imagem

## Personalização

Você pode ajustar vários parâmetros para personalizar o comportamento do framework:

- `frame_similarity_threshold`: Limiar para considerar um frame como principal (padrão: 0.8)
- `frame_interval`: Intervalo em segundos para extrair frames (padrão: 1.0)
- `output_dir`: Diretório para armazenar os dados processados (padrão: "processed_data")
- `db_path`: Caminho personalizado para o banco de dados SQLite
- `index_path`: Caminho personalizado para o índice FAISS
- `semantic_weight`: Peso para a busca semântica na busca por texto (0-1, padrão: 0.5)
- Modelos utilizados: você pode alterar os modelos em `image_models.py` e `text_models.py`

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests.

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes.