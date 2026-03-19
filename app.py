import os
import sqlite3
from flask import Flask, request, render_template, jsonify
import feedparser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import google.generativeai as genai

app = Flask(__name__)
DB_PATH = os.path.join(os.path.dirname(__file__), 'news_judge.db')

# =========================
# 🔥 DB初期化（←ここ重要）
# =========================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS judgments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            news_sim_score REAL,
            ai_score REAL,
            final_score REAL,
            ai_analysis TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# 👇 Flask起動時に必ず実行させる
init_db()

# =========================
# Gemini API
# =========================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# =========================
# ニュース取得
# =========================
def get_news(url):
    try:
        feed = feedparser.parse(url)
        return [entry.title for entry in feed.entries[:20]]
    except:
        return []

# =========================
# 類似度計算
# =========================
def calc_similarity(input_text, news_list):
    if not news_list:
        return 0.0
    texts = [input_text] + news_list
    vectorizer = TfidfVectorizer()
    try:
        tfidf = vectorizer.fit_transform(texts)
        sim = cosine_similarity(tfidf[0:1], tfidf[1:])
        return float(np.max(sim[0]))
    except:
        return 0.0

# =========================
# AI判定
# =========================
def get_ai_judgment(text):
    if not GEMINI_API_KEY:
        return 50, "APIキー未設定"

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(f"""
        以下の文章の信頼性を100点満点で評価してください。
        ScoreとReasonの形式で答えてください。

        {text}
        """)

        content = response.text

        score = 50
        reason = "解析失敗"

        for line in content.split("\n"):
            if "Score:" in line:
                try:
                    score = float(line.split(":")[1].strip())
                except:
                    pass
            if "Reason:" in line:
                reason = line.split(":", 1)[1].strip()

        return score, reason

    except Exception as e:
        print("Gemini error:", e)
        return 50, "AIエラー"

# =========================
# スコア計算
# =========================
def calculate_final_score(news_sim, ai_score):
    news_score = min(100, news_sim * 250)
    return round((news_score * 0.4) + (ai_score * 0.6), 1)

# =========================
# DB保存
# =========================
def save_judgment(text, news_sim, ai_score, final_score, ai_analysis):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO judgments (text, news_sim_score, ai_score, final_score, ai_analysis)
        VALUES (?, ?, ?, ?, ?)
    ''', (text, news_sim, ai_score, final_score, ai_analysis))
    conn.commit()
    conn.close()

# =========================
# 履歴取得
# =========================
def get_history(limit=10):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            SELECT text, final_score, created_at 
            FROM judgments 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (limit,))
        history = cursor.fetchall()
    except:
        history = []
    conn.close()
    return history

# =========================
# ルート
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    input_text = ""

    if request.method == "POST":
        input_text = request.form.get("text", "")
        if input_text.strip():
            yahoo = get_news("https://news.yahoo.co.jp/rss/topics/top-picks.xml")
            google = get_news("https://news.google.com/rss?hl=ja&gl=JP&ceid=JP:ja")

            news_sim = max(
                calc_similarity(input_text, yahoo),
                calc_similarity(input_text, google)
            )

            ai_score, ai_analysis = get_ai_judgment(input_text)
            final_score = calculate_final_score(news_sim, ai_score)

            save_judgment(input_text, news_sim, ai_score, final_score, ai_analysis)

            result = {
                "score": final_score,
                "analysis": ai_analysis
            }

    history = get_history()
    return render_template("index.html", result=result, input_text=input_text, history=history)

# =========================
# API
# =========================
@app.route("/api/judge", methods=["POST"])
def api_judge():
    data = request.json

    if not data or 'text' not in data:
        return jsonify({"error": "No text"}), 400

    text = data['text']

    yahoo = get_news("https://news.yahoo.co.jp/rss/topics/top-picks.xml")
    google = get_news("https://news.google.com/rss?hl=ja&gl=JP&ceid=JP:ja")

    news_sim = max(
        calc_similarity(text, yahoo),
        calc_similarity(text, google)
    )

    ai_score, ai_analysis = get_ai_judgment(text)
    final_score = calculate_final_score(news_sim, ai_score)

    save_judgment(text, news_sim, ai_score, final_score, ai_analysis)

    return jsonify({
        "score": final_score,
        "ai_score": ai_score,
        "analysis": ai_analysis
    })

# =========================
# 起動
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
