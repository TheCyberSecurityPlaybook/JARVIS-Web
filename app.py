#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JARVIS WEB — Локальный ИИ-помощник с футуристическим интерфейсом
Порт: 8001 (исправлена обработка потоковых сообщений)
"""

import os
import json
import hashlib
import asyncio
import httpx
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path
from functools import lru_cache

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import HTMLResponse
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import PyPDF2

# ===== НАСТРОЙКИ =====
OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:3b"          # лёгкая и быстрая (можно сменить)
KNOWLEDGE_BASE_PATH = "./knowledge_base"
VECTOR_DB_PATH = "./chroma_db"
STATE_FILE = "./index_state.json"
UPDATE_INTERVAL_HOURS = 24
CACHE_SIZE = 100
USE_KNOWLEDGE_BASE = True                # можно временно отключить для отладки

app = FastAPI(title="JARVIS Web")

# ===== КЛАСС БАЗЫ ЗНАНИЙ =====
class KnowledgeBase:
    """Управление векторной базой знаний с инкрементальным обновлением"""
    def __init__(self, kb_path: str, db_path: str, state_file: str):
        self.kb_path = Path(kb_path)
        self.db_path = db_path
        self.state_file = state_file
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.kb_path.mkdir(exist_ok=True, parents=True)
        os.makedirs(db_path, exist_ok=True)

        # Отключаем телеметрию ChromaDB
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {"files": {}, "last_update": None}
        return {"files": {}, "last_update": None}

    def _save_state(self):
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

    def _get_file_hash(self, file_path: Path) -> str:
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for block in iter(lambda: f.read(65536), b''):
                sha256.update(block)
        return sha256.hexdigest()

    def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Ошибка при чтении PDF {pdf_path}: {e}")
        return text

    def _extract_text_from_txt(self, txt_path: Path) -> str:
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(txt_path, 'r', encoding='cp1251') as f:
                    return f.read()
            except:
                return ""

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        return chunks

    def _process_file(self, file_path: Path) -> List[Dict]:
        chunks = []
        if file_path.suffix.lower() == '.pdf':
            text = self._extract_text_from_pdf(file_path)
        elif file_path.suffix.lower() in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json']:
            text = self._extract_text_from_txt(file_path)
        else:
            return []
        if not text.strip():
            return []
        text_chunks = self._chunk_text(text)
        for idx, chunk in enumerate(text_chunks):
            chunks.append({
                "text": chunk,
                "metadata": {
                    "file_path": str(file_path.relative_to(self.kb_path)),
                    "file_name": file_path.name,
                    "chunk_index": idx,
                    "total_chunks": len(text_chunks)
                }
            })
        return chunks

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self.embedding_model.encode(texts, convert_to_numpy=True).tolist()

    def update_index(self, full_rebuild: bool = False) -> Dict:
        start_time = datetime.now()
        all_files = []
        for ext in ['*.txt', '*.md', '*.pdf', '*.py', '*.js', '*.html', '*.css', '*.json']:
            all_files.extend(self.kb_path.rglob(ext))

        current_hashes = {}
        changed_files = []
        deleted_files = []

        for file_path in all_files:
            rel_path = str(file_path.relative_to(self.kb_path))
            file_hash = self._get_file_hash(file_path)
            current_hashes[rel_path] = file_hash
            if rel_path not in self.state["files"] or self.state["files"][rel_path] != file_hash:
                changed_files.append(file_path)

        for rel_path in list(self.state["files"].keys()):
            if not (self.kb_path / rel_path).exists():
                deleted_files.append(rel_path)

        print(f"🔍 Найдено изменений: {len(changed_files)} новых/изменённых, {len(deleted_files)} удалённых")

        if deleted_files and not full_rebuild:
            for rel_path in deleted_files:
                results = self.collection.get(where={"file_path": rel_path})
                if results and results.get("ids"):
                    self.collection.delete(ids=results["ids"])
                del self.state["files"][rel_path]
            print(f"🗑️ Удалено {len(deleted_files)} файлов из индекса")

        if full_rebuild:
            existing = self.collection.get()
            if existing and existing.get("ids"):
                self.collection.delete(ids=existing["ids"])
            self.state["files"] = {}
            changed_files = all_files
            print("🔄 Полная перестройка индекса")

        if changed_files:
            all_chunks, all_ids, all_metadatas = [], [], []
            for file_path in changed_files:
                rel_path = str(file_path.relative_to(self.kb_path))
                if not full_rebuild and rel_path in self.state["files"]:
                    results = self.collection.get(where={"file_path": rel_path})
                    if results and results.get("ids"):
                        self.collection.delete(ids=results["ids"])
                chunks = self._process_file(file_path)
                for chunk in chunks:
                    chunk_id = f"{rel_path}::{chunk['metadata']['chunk_index']}"
                    all_chunks.append(chunk["text"])
                    all_ids.append(chunk_id)
                    all_metadatas.append(chunk["metadata"])

            if all_chunks:
                batch_size = 32
                for i in range(0, len(all_chunks), batch_size):
                    batch_chunks = all_chunks[i:i+batch_size]
                    batch_ids = all_ids[i:i+batch_size]
                    batch_metadatas = all_metadatas[i:i+batch_size]
                    batch_embeddings = self._get_embeddings(batch_chunks)
                    self.collection.add(
                        ids=batch_ids,
                        documents=batch_chunks,
                        embeddings=batch_embeddings,
                        metadatas=batch_metadatas
                    )
                for file_path in changed_files:
                    rel_path = str(file_path.relative_to(self.kb_path))
                    self.state["files"][rel_path] = current_hashes[rel_path]
                print(f"✅ Проиндексировано {len(all_chunks)} чанков из {len(changed_files)} файлов")

        self.state["last_update"] = datetime.now().isoformat()
        self._save_state()
        elapsed = (datetime.now() - start_time).total_seconds()
        return {
            "status": "success",
            "files_changed": len(changed_files),
            "files_deleted": len(deleted_files),
            "total_files": len(all_files),
            "elapsed_seconds": round(elapsed, 2)
        }

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        try:
            query_embedding = self._get_embeddings([query])[0]
            results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
            search_results = []
            if results and results.get("documents") and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    search_results.append({
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                        "score": results["distances"][0][i] if results.get("distances") else 0
                    })
            return search_results
        except Exception as e:
            print(f"Ошибка при поиске в БЗ: {e}")
            return []

# ===== ИНИЦИАЛИЗАЦИЯ =====
kb = KnowledgeBase(KNOWLEDGE_BASE_PATH, VECTOR_DB_PATH, STATE_FILE)

scheduler = BackgroundScheduler()
scheduler.add_job(
    kb.update_index,
    trigger=IntervalTrigger(hours=UPDATE_INTERVAL_HOURS),
    id="update_knowledge_base",
    replace_existing=True
)
scheduler.start()

# ===== ВЕБ-ИНТЕРФЕЙС (ПОЛНЫЙ HTML) =====
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JARVIS Web - Киберпространственный интерфейс</title>
    <style>
        /* ===== СТИЛИ (те же, что и ранее, для краткости оставлены без изменений) ===== */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Share Tech Mono', 'Courier New', monospace;
            background: #0a0f1e;
            color: #e0e0ff;
            overflow: hidden;
            height: 100vh;
            position: relative;
        }
        #grid-canvas {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            opacity: 0.3;
            pointer-events: none;
        }
        .hologram-panel {
            position: relative;
            background: rgba(10, 20, 40, 0.7);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(0, 255, 255, 0.3);
            border-radius: 12px;
            box-shadow: 0 0 30px rgba(0, 255, 255, 0.2),
                        inset 0 0 20px rgba(0, 255, 255, 0.1);
            padding: 20px;
            margin: 15px;
            overflow: hidden;
        }
        .hologram-panel::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(45deg, transparent 30%, rgba(0, 255, 255, 0.1) 50%, transparent 70%);
            animation: scan 8s linear infinite;
            pointer-events: none;
        }
        @keyframes scan {
            0% { transform: translate(-30%, -30%) rotate(0deg); }
            100% { transform: translate(30%, 30%) rotate(0deg); }
        }
        .neon-text {
            color: #fff;
            text-shadow: 0 0 5px #00ffff, 0 0 10px #00ffff, 0 0 20px #00ffff, 0 0 40px #00ffff;
            font-weight: 400;
            letter-spacing: 2px;
        }
        .glitch {
            animation: glitch 3s infinite;
        }
        @keyframes glitch {
            0%,100% { transform: none; opacity: 1; }
            7% { transform: skew(-0.5deg, -0.9deg); opacity: 0.75; }
            10% { transform: none; opacity: 1; }
            30% { transform: skew(0.8deg, -0.1deg); opacity: 0.75; }
            35% { transform: none; opacity: 1; }
            55% { transform: skew(-1deg, 0.2deg); opacity: 0.75; }
            50% { transform: none; opacity: 1; }
            75% { transform: skew(0.4deg, 1deg); opacity: 0.75; }
            80% { transform: none; opacity: 1; }
        }
        .cyber-button {
            background: transparent;
            border: 1px solid #00ffff;
            color: #00ffff;
            padding: 8px 16px;
            font-family: 'Share Tech Mono', monospace;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 2px;
            cursor: pointer;
            position: relative;
            overflow: hidden;
            transition: all 0.3s;
            box-shadow: 0 0 10px rgba(0, 255, 255, 0.3);
            border-radius: 4px;
        }
        .cyber-button:hover {
            background: rgba(0, 255, 255, 0.1);
            box-shadow: 0 0 20px #00ffff;
            border-color: #00ffff;
            color: #ffffff;
            text-shadow: 0 0 5px #00ffff;
        }
        .cyber-button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(0, 255, 255, 0.2), transparent);
            transition: left 0.5s;
        }
        .cyber-button:hover::before {
            left: 100%;
        }
        .cyber-input {
            background: rgba(0, 20, 40, 0.8);
            border: 1px solid #00ffff;
            color: #00ffff;
            padding: 10px 15px;
            font-family: 'Share Tech Mono', monospace;
            width: 100%;
            border-radius: 4px;
            box-shadow: inset 0 0 10px rgba(0, 255, 255, 0.2);
            transition: all 0.3s;
        }
        .cyber-input:focus {
            outline: none;
            box-shadow: 0 0 20px rgba(0, 255, 255, 0.5), inset 0 0 10px rgba(0, 255, 255, 0.3);
            border-color: #ffffff;
        }
        .cyber-select {
            background: rgba(0, 20, 40, 0.8);
            border: 1px solid #00ffff;
            color: #00ffff;
            padding: 8px;
            font-family: 'Share Tech Mono', monospace;
            border-radius: 4px;
            width: 100%;
            cursor: pointer;
        }
        .cyber-select option {
            background: #0a0f1e;
            color: #00ffff;
        }
        #chat-messages {
            height: 500px;
            overflow-y: auto;
            padding: 15px;
            background: rgba(0, 10, 20, 0.6);
            border-radius: 8px;
            margin-bottom: 15px;
            border: 1px solid rgba(0, 255, 255, 0.2);
            scroll-behavior: smooth;
        }
        .message {
            margin-bottom: 15px;
            padding: 10px 15px;
            border-radius: 8px;
            animation: messageAppear 0.3s ease-out;
        }
        @keyframes messageAppear {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .user-message {
            background: rgba(0, 100, 255, 0.2);
            border-left: 3px solid #00ffff;
            margin-left: 20%;
        }
        .assistant-message {
            background: rgba(255, 0, 255, 0.1);
            border-left: 3px solid #ff00ff;
            margin-right: 20%;
        }
        .message-header {
            font-size: 12px;
            color: rgba(255, 255, 255, 0.6);
            margin-bottom: 5px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .message-content {
            font-size: 14px;
            line-height: 1.6;
            white-space: pre-wrap;
        }
        .message-content pre {
            background: rgba(0, 0, 0, 0.5);
            padding: 10px;
            border-radius: 4px;
            border: 1px solid #00ffff;
            overflow-x: auto;
        }
        .message-content code {
            background: rgba(0, 255, 255, 0.1);
            padding: 2px 4px;
            border-radius: 2px;
            color: #ff00ff;
        }
        .status-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 15px;
            background: rgba(0, 0, 0, 0.5);
            border-bottom: 1px solid #00ffff;
            font-size: 12px;
            color: #00ffff;
        }
        .led {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }
        .led-green {
            background: #00ff00;
            box-shadow: 0 0 10px #00ff00;
            animation: pulse 2s infinite;
        }
        .led-blue {
            background: #00ffff;
            box-shadow: 0 0 10px #00ffff;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        .typing-indicator {
            display: flex;
            padding: 10px;
        }
        .typing-indicator span {
            width: 8px;
            height: 8px;
            background: #00ffff;
            border-radius: 50%;
            margin: 0 2px;
            animation: typing 1.4s infinite;
            box-shadow: 0 0 10px #00ffff;
        }
        .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
        .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes typing {
            0%,60%,100% { transform: translateY(0); opacity: 0.6; }
            30% { transform: translateY(-10px); opacity: 1; }
        }
        .container {
            display: grid;
            grid-template-columns: 300px 1fr;
            height: 100vh;
            gap: 10px;
            padding: 10px;
        }
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        .main {
            display: flex;
            flex-direction: column;
        }
        @media (max-width: 768px) {
            .container { grid-template-columns: 1fr; }
            .sidebar { display: none; }
        }
    </style>
</head>
<body>
    <canvas id="grid-canvas"></canvas>

    <div class="status-bar">
        <div><span class="led led-green"></span> JARVIS ONLINE</div>
        <div><span class="led led-blue"></span> SECURE MODE</div>
        <div id="model-status">МОДЕЛЬ: {{ default_model }}</div>
        <div id="kb-status">БАЗА ЗНАНИЙ: {{ kb_files }} файлов</div>
    </div>

    <div class="container">
        <!-- Боковая панель -->
        <div class="sidebar">
            <div class="hologram-panel">
                <h2 class="neon-text glitch" style="font-size: 18px; margin-bottom: 15px;">⚡ СИСТЕМА УПРАВЛЕНИЯ</h2>
                <div style="margin-bottom: 15px;">
                    <label style="color: #00ffff; font-size: 12px;">МОДЕЛЬ</label>
                    <select id="model-select" class="cyber-select"></select>
                </div>
                <div style="margin-bottom: 15px;">
                    <label style="color: #00ffff; font-size: 12px;">ТЕМПЕРАТУРА</label>
                    <input type="range" id="temperature" min="0" max="2" step="0.1" value="0.7" class="cyber-input" style="padding: 0;">
                    <span id="temp-value" style="color: #ff00ff; float: right;">0.7</span>
                </div>
                <div style="margin-bottom: 15px;">
                    <label style="color: #00ffff; font-size: 12px;">MAX TOKENS</label>
                    <input type="number" id="max-tokens" value="2048" min="128" max="8192" class="cyber-input">
                </div>
                <div style="margin-bottom: 15px;">
                    <label style="color: #00ffff; font-size: 12px;">СИСТЕМНЫЙ ПРОМПТ</label>
                    <textarea id="system-prompt" rows="3" class="cyber-input">Ты — JARVIS, помощник для программирования и кибербезопасности. Отвечай кратко и по делу.</textarea>
                </div>
                <button id="new-chat" class="cyber-button" style="width: 100%; margin-bottom: 10px;">🆕 НОВЫЙ ЧАТ</button>
                <button id="update-kb" class="cyber-button" style="width: 100%;">🔄 ОБНОВИТЬ ЗНАНИЯ</button>
            </div>
            <div class="hologram-panel">
                <h2 class="neon-text" style="font-size: 16px; margin-bottom: 10px;">📁 БАЗА ЗНАНИЙ</h2>
                <div id="kb-info">
                    <p style="color: #00ffff; font-size: 12px;">Файлов: {{ kb_files }}<br>Последнее обновление: {{ last_update }}</p>
                </div>
                <div style="max-height: 200px; overflow-y: auto; margin-top: 10px;">
                    <ul id="kb-files-list" style="list-style: none; color: #ff00ff; font-size: 11px;"></ul>
                </div>
            </div>
        </div>

        <!-- Основная область чата -->
        <div class="main">
            <div class="hologram-panel" style="flex: 1; display: flex; flex-direction: column;">
                <div id="chat-messages">
                    <div class="message assistant-message">
                        <div class="message-header">🤖 JARVIS v2.0</div>
                        <div class="message-content">Добро пожаловать, агент. Система готова к работе.<br>Доступные модели: <span id="model-list-placeholder">загрузка...</span></div>
                    </div>
                </div>
                <div style="display: flex; gap: 10px; margin-top: 10px;">
                    <input type="text" id="user-input" class="cyber-input" placeholder="ВВЕДИТЕ СООБЩЕНИЕ..." autofocus>
                    <button id="send-button" class="cyber-button">ОТПРАВИТЬ</button>
                </div>
                <div style="display: flex; gap: 10px; margin-top: 10px; font-size: 11px; color: #6666ff;">
                    <span>⚡ WebSocket: <span id="ws-status">ПОДКЛЮЧЕНИЕ...</span></span>
                    <span>🔒 Шифрование: AES-256</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('grid-canvas');
        const ctx = canvas.getContext('2d');
        function resizeCanvas() { canvas.width = window.innerWidth; canvas.height = window.innerHeight; }
        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();
        function drawGrid() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const cellSize = 50, time = Date.now() / 1000;
            ctx.strokeStyle = '#00ffff'; ctx.lineWidth = 0.5; ctx.globalAlpha = 0.2;
            for (let x = 0; x < canvas.width; x += cellSize) {
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x + Math.sin(time) * 5, canvas.height);
                ctx.strokeStyle = `rgba(0, 255, 255, ${0.1 + Math.sin(x + time) * 0.1})`;
                ctx.stroke();
            }
            for (let y = 0; y < canvas.height; y += cellSize) {
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(canvas.width, y + Math.cos(time) * 5);
                ctx.strokeStyle = `rgba(255, 0, 255, ${0.1 + Math.cos(y + time) * 0.1})`;
                ctx.stroke();
            }
            ctx.fillStyle = '#00ffff'; ctx.globalAlpha = 0.5;
            for (let x = 0; x < canvas.width; x += cellSize * 2)
                for (let y = 0; y < canvas.height; y += cellSize * 2) {
                    const pulse = Math.sin(x + y + time * 2) * 0.5 + 0.5;
                    ctx.globalAlpha = pulse * 0.3;
                    ctx.beginPath();
                    ctx.arc(x + Math.sin(time) * 10, y + Math.cos(time) * 10, 2, 0, Math.PI * 2);
                    ctx.fill();
                }
            requestAnimationFrame(drawGrid);
        }
        drawGrid();

        let ws = null;
        let isTyping = false;
        let currentAssistantMessage = null; // для потокового вывода

        const chatMessages = document.getElementById('chat-messages');
        const userInput = document.getElementById('user-input');
        const sendButton = document.getElementById('send-button');
        const modelSelect = document.getElementById('model-select');
        const temperatureInput = document.getElementById('temperature');
        const tempValue = document.getElementById('temp-value');
        const maxTokensInput = document.getElementById('max-tokens');
        const systemPromptInput = document.getElementById('system-prompt');
        const wsStatus = document.getElementById('ws-status');

        async function loadModels() {
            try {
                const response = await fetch('/api/models');
                const models = await response.json();
                modelSelect.innerHTML = '';
                models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.name;
                    option.textContent = model.name;
                    if (model.name === '{{ default_model }}') option.selected = true;
                    modelSelect.appendChild(option);
                });
                document.getElementById('model-list-placeholder').textContent = models.map(m => m.name).join(', ');
            } catch (error) { console.error('Ошибка загрузки моделей:', error); }
        }

        async function loadKB() {
            try {
                const response = await fetch('/api/kb/files');
                const files = await response.json();
                const filesList = document.getElementById('kb-files-list');
                filesList.innerHTML = '';
                files.slice(0, 10).forEach(file => {
                    const li = document.createElement('li');
                    li.textContent = `📄 ${file}`;
                    li.style.marginBottom = '5px';
                    filesList.appendChild(li);
                });
                if (files.length > 10) {
                    const li = document.createElement('li');
                    li.textContent = `... и еще ${files.length - 10} файлов`;
                    li.style.color = '#6666ff';
                    filesList.appendChild(li);
                }
            } catch (error) { console.error('Ошибка загрузки KB:', error); }
        }

        function connectWebSocket() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);
            ws.onopen = () => {
                wsStatus.textContent = '🟢 ONLINE';
                wsStatus.style.color = '#00ff00';
            };
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                console.log('Received:', data);

                if (data.type === 'chunk') {
                    if (!currentAssistantMessage) {
                        // Создаём новое сообщение ассистента
                        const messageDiv = document.createElement('div');
                        messageDiv.className = 'message assistant-message';
                        const header = document.createElement('div');
                        header.className = 'message-header';
                        header.textContent = '🤖 JARVIS';
                        const contentDiv = document.createElement('div');
                        contentDiv.className = 'message-content';
                        messageDiv.appendChild(header);
                        messageDiv.appendChild(contentDiv);
                        chatMessages.appendChild(messageDiv);
                        currentAssistantMessage = contentDiv;
                    }
                    // Дописываем контент
                    currentAssistantMessage.innerHTML += data.content;
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                } else if (data.type === 'done') {
                    currentAssistantMessage = null;
                    isTyping = false;
                    document.querySelectorAll('.typing-indicator').forEach(el => el.remove());
                } else if (data.type === 'error') {
                    addMessage('assistant', `⚠️ Ошибка: ${data.content}`);
                    isTyping = false;
                    document.querySelectorAll('.typing-indicator').forEach(el => el.remove());
                    currentAssistantMessage = null;
                }
            };
            ws.onclose = () => {
                wsStatus.textContent = '🔴 OFFLINE';
                wsStatus.style.color = '#ff0000';
                setTimeout(connectWebSocket, 3000);
            };
            ws.onerror = (error) => console.error('WebSocket error:', error);
        }

        function addMessage(role, content) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role === 'user' ? 'user-message' : 'assistant-message'}`;
            const header = document.createElement('div');
            header.className = 'message-header';
            header.textContent = role === 'user' ? '👤 АГЕНТ' : '🤖 JARVIS';
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            // Простая обработка markdown-подобного форматирования
            content = content.replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
            content = content.replace(/`([^`]+)`/g, '<code>$1</code>');
            content = content.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
            content = content.replace(/\*(.+?)\*/g, '<em>$1</em>');
            content = content.replace(/\n/g, '<br>');
            contentDiv.innerHTML = content;
            messageDiv.appendChild(header);
            messageDiv.appendChild(contentDiv);
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        function sendMessage() {
            const message = userInput.value.trim();
            if (!message || isTyping) return;

            addMessage('user', message);
            userInput.value = '';

            // Показываем индикатор печати (можно оставить, он не мешает)
            const indicator = document.createElement('div');
            indicator.className = 'message assistant-message typing-indicator';
            indicator.innerHTML = '<span></span><span></span><span></span>';
            chatMessages.appendChild(indicator);
            chatMessages.scrollTop = chatMessages.scrollHeight;
            isTyping = true;

            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    message: message,
                    model: modelSelect.value,
                    temperature: parseFloat(temperatureInput.value),
                    max_tokens: parseInt(maxTokensInput.value),
                    system_prompt: systemPromptInput.value,
                    use_kb: true
                }));
            } else {
                addMessage('assistant', '⚠️ Ошибка соединения. Переподключение...');
                isTyping = false;
                document.querySelectorAll('.typing-indicator').forEach(el => el.remove());
            }
        }

        sendButton.addEventListener('click', sendMessage);
        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
        });
        temperatureInput.addEventListener('input', () => { tempValue.textContent = temperatureInput.value; });
        document.getElementById('new-chat').addEventListener('click', () => {
            chatMessages.innerHTML = `
                <div class="message assistant-message">
                    <div class="message-header">🤖 JARVIS v2.0</div>
                    <div class="message-content">Чат очищен. Начните новый разговор.</div>
                </div>
            `;
            currentAssistantMessage = null;
        });
        document.getElementById('update-kb').addEventListener('click', async () => {
            try {
                const response = await fetch('/api/kb/update', { method: 'POST' });
                const result = await response.json();
                addMessage('assistant', `✅ База знаний обновлена: ${result.files_changed} файлов изменено, ${result.files_deleted} удалено за ${result.elapsed_seconds}с`);
                loadKB();
            } catch (error) {
                addMessage('assistant', `❌ Ошибка обновления БЗ: ${error}`);
            }
        });

        loadModels();
        loadKB();
        connectWebSocket();
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    kb_files = []
    if os.path.exists(KNOWLEDGE_BASE_PATH):
        for ext in ['*.txt', '*.md', '*.pdf', '*.py', '*.js', '*.html', '*.css', '*.json']:
            kb_files.extend([f.name for f in Path(KNOWLEDGE_BASE_PATH).rglob(ext)])

    last_update = kb.state.get("last_update", "никогда")
    if last_update and last_update != "никогда":
        try:
            last_update = datetime.fromisoformat(last_update).strftime("%Y-%m-%d %H:%M")
        except:
            pass

    html = HTML_TEMPLATE.replace("{{ default_model }}", DEFAULT_MODEL)\
                        .replace("{{ kb_files }}", str(len(kb_files)))\
                        .replace("{{ last_update }}", str(last_update))
    return HTMLResponse(content=html)

# ===== API =====
@app.get("/api/models")
async def get_models():
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{OLLAMA_HOST}/api/tags", timeout=5.0)
            if resp.status_code == 200:
                return resp.json().get("models", [])
    except Exception as e:
        print(f"Ошибка получения моделей: {e}")
    return []

@app.get("/api/kb/files")
async def get_kb_files():
    files = []
    if os.path.exists(KNOWLEDGE_BASE_PATH):
        for ext in ['*.txt', '*.md', '*.pdf', '*.py', '*.js', '*.html', '*.css', '*.json']:
            for f in Path(KNOWLEDGE_BASE_PATH).rglob(ext):
                files.append(str(f.relative_to(KNOWLEDGE_BASE_PATH)))
    return files

@app.post("/api/kb/update")
async def update_knowledge_base(background_tasks: BackgroundTasks):
    return kb.update_index()

@app.post("/api/kb/rebuild")
async def rebuild_knowledge_base(background_tasks: BackgroundTasks):
    return kb.update_index(full_rebuild=True)

@app.get("/api/kb/search")
async def search_kb(q: str, top_k: int = 5):
    return kb.search(q, top_k)

# ===== WEBSOCKET =====
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ WebSocket подключён")

    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("message", "")
            model = data.get("model", DEFAULT_MODEL)
            temperature = data.get("temperature", 0.7)
            max_tokens = data.get("max_tokens", 2048)
            system_prompt = data.get("system_prompt", "Ты — JARVIS, помощник для программирования и кибербезопасности. Отвечай кратко и по делу.")
            use_kb = data.get("use_kb", True) and USE_KNOWLEDGE_BASE

            print(f"📨 Запрос: {message[:50]}... (модель: {model})")

            context = ""
            if use_kb:
                try:
                    search_results = kb.search(message, top_k=3)
                    if search_results:
                        context = "Контекст из базы знаний:\n\n"
                        for i, r in enumerate(search_results):
                            context += f"[{i+1}] {r['metadata'].get('file_path', 'unknown')}:\n{r['text'][:300]}...\n\n"
                except Exception as e:
                    print(f"Ошибка при поиске в БЗ: {e}")

            full_prompt = message
            if context:
                full_prompt = f"{context}\nВопрос: {message}\n\nОтвет:"

            ollama_request = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ],
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }

            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    async with client.stream("POST", f"{OLLAMA_HOST}/api/chat", json=ollama_request) as response:
                        if response.status_code != 200:
                            error_text = await response.aread()
                            await websocket.send_json({"type": "error", "content": f"Ошибка Ollama: {error_text.decode()}"})
                            continue

                        async for line in response.aiter_lines():
                            if line:
                                try:
                                    chunk = json.loads(line)
                                    if chunk.get("message", {}).get("content"):
                                        await websocket.send_json({
                                            "type": "chunk",
                                            "content": chunk["message"]["content"]
                                        })
                                except json.JSONDecodeError:
                                    pass
            except Exception as e:
                await websocket.send_json({"type": "error", "content": f"Ошибка соединения: {str(e)}"})

            await websocket.send_json({"type": "done"})
            print("✅ Ответ отправлен")

    except WebSocketDisconnect:
        print("❌ WebSocket отключён")
    except Exception as e:
        print(f"⚠️ Ошибка WebSocket: {e}")

# ===== ЗАПУСК =====
if __name__ == "__main__":
    import uvicorn

    print("🔄 Первоначальное обновление базы знаний...")
    try:
        kb.update_index()
    except Exception as e:
        print(f"⚠️ Ошибка при первом обновлении: {e}")

    try:
        r = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        if r.status_code == 200:
            models = r.json().get("models", [])
            print(f"✅ Ollama доступна, найдено моделей: {len(models)}")
        else:
            print("⚠️ Ollama не отвечает корректно")
    except Exception as e:
        print(f"⚠️ Не удалось подключиться к Ollama: {e}")

    PORT = 8001
    print(f"🚀 Запуск JARVIS Web на http://localhost:{PORT}")
    print(f"📁 База знаний: {KNOWLEDGE_BASE_PATH}")
    print(f"🔄 Автообновление каждые {UPDATE_INTERVAL_HOURS} часов")
    print(f"📊 Использование БЗ: {USE_KNOWLEDGE_BASE}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
