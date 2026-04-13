from langchain_ollama import ChatOllama
import subprocess
import time

# Start ollama serve in background
process = subprocess.Popen(
    ['ollama', 'serve'],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)
time.sleep(3)  # wait for server to start

ollama_llm = ChatOllama(model='gemma3:4b', temperature=0.4)