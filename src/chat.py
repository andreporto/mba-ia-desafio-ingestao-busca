from search import search_prompt

def main():
    print("--- Bem-vindo ao Chat RAG ---")
    print("Digite 'sair' para encerrar.")
    
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return
    
    while True:
        try:
            pergunta = input("\nFaça sua pergunta: ")
            
            if pergunta.lower() in ["sair", "exit", "quit"]:
                print("Até logo!")
                break
            
            if not pergunta.strip():
                continue
                
            print("PERGUNTA:", pergunta)
            resposta = chain.invoke(pergunta)
            print("RESPOSTA:", resposta)
            
        except KeyboardInterrupt:
            print("\nAté logo!")
            break
        except Exception as e:
            print(f"Ocorreu um erro: {e}")

if __name__ == "__main__":
    main()
