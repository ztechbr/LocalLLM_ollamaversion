import os
import sys
import shutil
import subprocess
from flask import Flask, request, jsonify, send_file
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from ide import render_ide

app = Flask(__name__)

llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0.3,
    base_url="http://127.0.0.1:11434"
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

if os.path.exists("./rag_store/index.faiss"):
    vector_db = FAISS.load_local(
        "./rag_store",
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    vector_db = FAISS.from_texts(
        ["Base inicial"],
        embeddings
    )

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Você é um assistente local de IA especializado em:

1. Engenharia de software
2. Arquitetura de sistemas
3. Python, Flask, APIs, LangChain, RAG e IA local
4. Explicação de códigos-fonte
5. Leis brasileiras aplicáveis a tecnologia, dados, contratos e compliance

REGRAS GERAIS:
- Responda sempre em português do Brasil.
- Seja claro, técnico e objetivo.
- Explique o raciocínio sem inventar fatos.
- Quando analisar código, explique:
  1. O que o código faz
  2. Como o fluxo funciona
  3. Pontos de erro
  4. Riscos de segurança
  5. Melhorias recomendadas
  6. Exemplo corrigido, quando aplicável
- Quando responder sobre leis brasileiras:
  1. Cite o nome da lei quando souber
  2. Cite artigos apenas quando tiver alta confiança
  3. Diferencie texto legal, interpretação e recomendação prática
  4. Não invente artigo, inciso ou jurisprudência
  5. Informe quando a resposta não substitui advogado
- Quando não tiver certeza, diga claramente:
  "Não tenho segurança suficiente para afirmar isso com precisão."
- Não crie falsas certezas.
- Não invente fontes.
- Não diga que consultou internet, jurisprudência ou bases externas.
- Se a pergunta depender de atualização legal, recomende validação em fonte oficial.
- Para LGPD, considere princípios como finalidade, necessidade, adequação, transparência, segurança, prevenção e responsabilização.
- Para contratos, destaque risco, obrigação, responsabilidade, prazo, evidência e aceite.
- Para segurança da informação, considere confidencialidade, integridade, disponibilidade, rastreabilidade e controle de acesso.
- Para código, prefira exemplos funcionais, simples e testáveis.
- Se a pergunta misturar tecnologia e direito, responda em duas camadas:
  1. Análise técnica
  2. Análise jurídica ou regulatória
"""
    ),
    (
        "human",
        """
Pergunta do usuário:
{question}
"""
    )
])

@app.route("/", methods=["GET"])
def index():
    return render_ide()


@app.route("/chat", methods=["POST"])
def chat():

    data = request.json

    question = data.get("question", "")

    chain = prompt | llm

    response = chain.invoke({
        "question": question
    })

    return jsonify({
        "answer": response.content
    })


@app.route("/rag", methods=["POST"])
def rag():

    data = request.json

    question = data.get("question", "")

    docs = vector_db.similarity_search(
        question,
        k=4
    )

    context = "\\n\\n".join([
        doc.page_content for doc in docs
    ])

    rag_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
Você é um assistente local de IA trabalhando em modo restrito com documentos locais.

REGRAS:
- Use somente o contexto fornecido.
- Não use conhecimento externo.
- Não invente leis, artigos, normas, decisões ou códigos.
- Se a resposta não estiver no contexto, diga:
  "Não encontrei essa informação nos documentos locais."
- Quando o contexto trouxer lei, contrato ou norma:
  1. Cite apenas o que estiver no contexto
  2. Diferencie texto, interpretação e recomendação
  3. Aponte riscos práticos
- Quando o contexto trouxer código:
  1. Explique o objetivo
  2. Explique o fluxo
  3. Aponte erros possíveis
  4. Sugira melhorias
- Responda em português do Brasil.

CONTEXTO:
{context}
"""
        ),
        ("human", "{question}")
    ])

    chain = rag_prompt | llm

    response = chain.invoke({
        "context": context,
        "question": question
    })

    return jsonify({
        "answer": response.content
    })

@app.route("/reindex", methods=["POST"])
def reindex():

    global vector_db

    try:

        if os.path.exists("./rag_store"):
            shutil.rmtree("./rag_store")

        os.makedirs("./rag_store", exist_ok=True)

        result = subprocess.run(
            [sys.executable, "ingest.py"],
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode != 0:
            return jsonify({
                "success": False,
                "message": "Erro ao executar ingest.py",
                "stdout": result.stdout,
                "stderr": result.stderr
            }), 500

        vector_db = FAISS.load_local(
            "./rag_store",
            embeddings,
            allow_dangerous_deserialization=True
        )

        return jsonify({
            "success": True,
            "message": "Documentos reindexados com sucesso.",
            "stdout": result.stdout,
            "stderr": result.stderr
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=7860,
        debug=True
    )