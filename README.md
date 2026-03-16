# Desafio MBA Engenharia de Software com IA - Full Cycle

Este projeto implementa um sistema de RAG (Retrieval-Augmented Generation) para ingestão e busca semântica em documentos PDF, utilizando LangChain, Google Gemini e PostgreSQL com pgvector.

## Como executar a solução

### 1. Pré-requisitos
- Docker e Docker Compose instalados.
- Python 3.10 ou superior.
- Uma chave de API do Google Gemini.

### 2. Configuração do ambiente
Crie um arquivo `.env` na raiz do projeto seguindo o modelo abaixo:
```env
GOOGLE_API_KEY=sua_chave_aqui
GOOGLE_EMBEDDING_MODEL=models/gemini-embedding-001
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
PG_VECTOR_COLLECTION_NAME=rag_collection
PDF_PATH=document.pdf
```

### 3. Instalação das dependências
Crie um ambiente virtual e instale as dependências:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Iniciar a infraestrutura (Postgres + pgvector)
Inicie o banco de dados via Docker Compose:
```bash
docker-compose up -d
```

### 5. Ingestão do PDF
Processe o documento PDF para gerar os embeddings e armazená-los no banco vetorial:
```bash
python src/ingest.py
```

### 6. Executar o Chat CLI
Inicie a interface de chat para interagir com o documento:
```bash
python src/chat.py
```

## Estrutura do Projeto
- `src/ingest.py`: Script para leitura do PDF, divisão em chunks e armazenamento vetorial.
- `src/search.py`: Lógica de busca semântica e template de prompt RAG.
- `src/chat.py`: Interface de linha de comando para interação com o usuário.
- `docker-compose.yml`: Configuração do banco de dados PostgreSQL com a extensão pgvector.
