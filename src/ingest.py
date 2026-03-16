import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
PG_VECTOR_COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME")
PDF_PATH = os.getenv("PDF_PATH")

def ingest_pdf():
    if not PDF_PATH or not os.path.exists(PDF_PATH):
        print(f"Erro: PDF_PATH não encontrado: {PDF_PATH}")
        return

    print(f"Carregando PDF de {PDF_PATH}...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    print(f"Dividindo {len(documents)} páginas em pedaços...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)

    print(f"Criando embeddings e armazenando {len(chunks)} pedaços no PostgreSQL...")
    embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv("GOOGLE_EMBEDDING_MODEL"))
    
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=PG_VECTOR_COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )

    # Batch processing to avoid rate limits
    batch_size = 1
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        print(f"Processando batch {i//batch_size + 1}/{len(chunks)}...")
        try:
            vectorstore.add_documents(batch)
        except Exception as e:
            if "429" in str(e):
                print("Limite de taxa atingido, dormindo por 30 segundos...")
                import time
                time.sleep(30)
                vectorstore.add_documents(batch)
            else:
                raise e
        import time
        time.sleep(5) # Sleep for 5 seconds between batches

    print("Ingestão concluída com sucesso.")


if __name__ == "__main__":
    ingest_pdf()
