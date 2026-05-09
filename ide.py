from flask import render_template_string


def render_ide():
    return render_template_string("""
<!DOCTYPE html>
<html lang="pt-BR">

<head>
    <meta charset="UTF-8">
    <title>Local LLM Chat</title>

    <style>
        body {
            background: #111827;
            color: #f9fafb;
            font-family: Arial;
            margin: 0;
            padding: 40px;
        }

        .container {
            width: 80%;
            margin: auto;
            background: #1f2937;
            padding: 30px;
            border-radius: 18px;
        }

        textarea {
            width: 80%;
            height: 80px;
            border-radius: 12px;
            border: 3px solid #374151;
            background: #111827;
            color: white;
            padding: 14px;
            font-size: 16px;
            resize: vertical;
        }

        button {
            margin-top: 15px;
            background: #2563eb;
            color: white;
            border: none;
            padding: 14px 20px;
            border-radius: 12px;
            cursor: pointer;
            font-size: 16px;
        }

        button:hover {
            opacity: 0.9;
        }

        .btn-tts {
            background: #16a34a;
            margin-left: 10px;
        }

        .btn-stop {
            background: #ea580c;
            margin-left: 10px;
        }

        .btn-danger {
            background: #dc2626;
            margin-left: 10px;
        }

        .response {
            margin-top: 30px;
            background: #030712;
            padding: 20px;
            border-radius: 12px;
            white-space: pre-wrap;
            line-height: 1.6;
            min-height: 120px;
        }

        .mode-box {
            margin: 20px 0;
            padding: 16px;
            border-radius: 12px;
            background: #111827;
            border: 1px solid #374151;
        }

        .mode-row {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .status {
            font-weight: bold;
        }

        .hint {
            color: #d1d5db;
            font-size: 14px;
            margin-top: 8px;
        }

        .spinner {
            display: none;
            margin-top: 25px;
            text-align: center;
        }

        .spinner-icon {
            width: 60px;
            height: 60px;
            border: 6px solid #374151;
            border-top: 6px solid #60a5fa;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: auto;
        }

        .spinner-text {
            margin-top: 12px;
            color: #93c5fd;
            font-size: 14px;
        }

        @keyframes spin {
            0% {
                transform: rotate(0deg);
            }

            100% {
                transform: rotate(360deg);
            }
        }
    </style>
</head>

<body>

    <div class="container">

        <h1>Local LLM Chat</h1>

        <p>Ollama + LangChain + Flask + FAISS + Browser TTS</p>

        <div class="mode-box">
            <div class="mode-row">
                <input
                    type="checkbox"
                    id="internetToggle"
                    checked
                    onchange="updateModeText()">

                <span id="modeStatus" class="status">
                    Modo com conhecimento geral habilitado
                </span>
            </div>

            <div id="modeHint" class="hint">
                O modelo usa o conhecimento interno do LLM.
            </div>
        </div>

        <textarea
            id="question"
            placeholder="Digite sua pergunta..."></textarea>

        <br>

        <div style="margin-top: 20px;">

            <button onclick="sendQuestion()">
                Enviar pergunta
            </button>

            <button onclick="playLastAnswerTTS()" class="btn-tts">
                Ouvir resposta
            </button>

            <button onclick="stopAudio()" class="btn-stop">
                Parar áudio
            </button>

            <button onclick="reindexDocuments()" class="btn-danger">
                Reindexar documentos
            </button>

        </div>

        <div id="loading" class="spinner">
            <div class="spinner-icon"></div>

            <div class="spinner-text">
                Aguarde, processando IA...
            </div>
        </div>

        <div id="response" class="response">
            A resposta aparecerá aqui.
        </div>

    </div>

    <script>
        let lastAnswerText = "";

        function updateModeText() {
            const internetEnabled =
                document.getElementById("internetToggle").checked;

            const modeStatus =
                document.getElementById("modeStatus");

            const modeHint =
                document.getElementById("modeHint");

            if (internetEnabled) {
                modeStatus.innerText =
                    "Modo com conhecimento geral habilitado";

                modeHint.innerText =
                    "O modelo usa o conhecimento interno do LLM.";
            } else {
                modeStatus.innerText =
                    "Modo somente documentos locais";

                modeHint.innerText =
                    "O modelo responderá somente usando o conteúdo indexado no FAISS.";
            }
        }

        function speakText(text) {
            if (!text || !text.trim()) {
                return;
            }

            window.speechSynthesis.cancel();

            const utterance =
                new SpeechSynthesisUtterance(text);

            utterance.lang = "pt-BR";
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            utterance.volume = 1.0;

            const voices =
                window.speechSynthesis.getVoices();

            const brazilVoice =
                voices.find(v =>
                    v.lang &&
                    v.lang.toLowerCase().includes("pt")
                );

            if (brazilVoice) {
                utterance.voice = brazilVoice;
            }

            window.speechSynthesis.speak(utterance);
        }

        function playLastAnswerTTS() {
            if (!lastAnswerText) {
                alert("Nenhuma resposta disponível para leitura.");
                return;
            }

            speakText(lastAnswerText);
        }

        function stopAudio() {
            window.speechSynthesis.cancel();
        }

        async function sendQuestion() {
            const question =
                document.getElementById("question").value;

            const internetEnabled =
                document.getElementById("internetToggle").checked;

            const endpoint =
                internetEnabled ? "/chat" : "/rag";

            const responseDiv =
                document.getElementById("response");

            const loading =
                document.getElementById("loading");

            if (!question.trim()) {
                responseDiv.innerText =
                    "Digite uma pergunta.";

                return;
            }

            loading.style.display = "block";
            responseDiv.innerText = "";
            lastAnswerText = "";

            window.speechSynthesis.cancel();

            try {
                const response =
                    await fetch(endpoint, {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json"
                        },
                        body: JSON.stringify({
                            question: question
                        })
                    });

                const data =
                    await response.json();

                const answerText =
                    data.answer
                    ? data.answer
                    : JSON.stringify(data, null, 2);

                responseDiv.innerText =
                    answerText;

                lastAnswerText =
                    answerText;

                speakText(answerText);

            } catch (error) {
                responseDiv.innerText =
                    "Erro: " + error;

            } finally {
                loading.style.display = "none";
            }
        }

        async function reindexDocuments() {
            const responseDiv =
                document.getElementById("response");

            const loading =
                document.getElementById("loading");

            loading.style.display = "block";

            responseDiv.innerText =
                "Reindexando documentos...";

            try {
                const response =
                    await fetch("/reindex", {
                        method: "POST"
                    });

                const data =
                    await response.json();

                responseDiv.innerText =
                    JSON.stringify(data, null, 2);

            } catch (error) {
                responseDiv.innerText =
                    "Erro ao chamar /reindex. Verifique o terminal do Flask.\\n\\n"
                    + error;

            } finally {
                loading.style.display = "none";
            }
        }

        window.speechSynthesis.onvoiceschanged =
            () => {};
    </script>

</body>

</html>
    """)