from flask import Flask, render_template, request
import sqlite3
import os
import google.generativeai as genai
import feedparser
import re
import time

app = Flask(__name__)

# =========================
# RSS（増やしてOK）
# =========================
RSS_URLS = [
    "https://www3.nhk.or.jp/rss/news/cat0.xml",
    "https://news.yahoo.co.jp/rss/topics/top-picks.xml",
    "http://feeds.bbci.co.uk/news/rss.xml"
]

# キャッシュ（超重要）
news_cache = {
    "data": [],
    "last_fetch": 0
}

CACHE_TTL = 300  # 5分

# =========================
# DB初期化
# =========================
def init_db():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS judgments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT,
        final_score REAL,
        ai_score REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

init_db()

# =========================
# Gemini設定
# =========================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = os.environ.get("GEMINI_MODEL", "models/gemini-2.5-flash")

model = None
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)

# =========================
# RSS取得（キャッシュ付き）
# =========================
def fetch_news():
    now = time.time()

    # キャッシュ使う
    if now - news_cache["last_fetch"] < CACHE_TTL:
        return news_cache["data"]

    articles = []

    for url in RSS_URLS:
        try:
            feed = feedparser.parse(url)
        except:
            continue

        if not feed.entries:
            continue

        for entry in feed.entries[:5]:  # ←軽量化（10→5）
            text = (entry.title + " " + entry.get("summary", "")).replace("\n", " ")

            articles.append({
                "title": entry.title,
                "link": entry.link,
                "text": text
            })

    # キャッシュ保存
    news_cache["data"] = articles
    news_cache["last_fetch"] = now

    return articles

# =========================
# 類似度（軽量版）
# =========================
def calc_similarity(input_text, articles):
    if not articles:
        return 0, []

    input_words = set(input_text.lower().split())
    scores = []

    for a in articles:
        article_words = set(a["text"].lower().split())
        score = len(input_words & article_words) / max(len(input_words), 1)
        scores.append(score)

    max_sim = max(scores) if scores else 0

    top_articles = [
        {
            "title": articles[i]["title"],
            "link": articles[i]["link"],
            "score": round(scores[i] * 100, 1)
        }
        for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        if scores[i] > 0.05
    ][:3]

    return max_sim * 100, top_articles

# =========================
# AI判定
# =========================
def get_ai_score_with_context(text, articles):
    if not model:
        return 0, "AI未使用"

    try:
        articles_text = "\n".join([
            f"{a['title']} ({a['score']}%)"
            for a in articles
        ]) if articles else "なし"

        prompt = f"""
以下の文章の信頼性を100点満点で評価してください。

参考ニュース:
{articles_text}

文章:
{text}

形式:
Score: 数値
Reason: 理由
"""

        response = model.generate_content(prompt)

        if not response or not response.text:
            return 0, "AIレスポンスなし"

        content = response.text

        match = re.search(r'(\d{1,3})', content)
        score = float(match.group(1)) if match else 0
        score = min(score, 100)

        return score, content.strip()

    except Exception as e:
        return 0, f"AIエラー: {str(e)}"

# =========================
# 総合判定
# =========================
def final_judgement(text):
    articles = fetch_news()

    sim_score, top_articles = calc_similarity(text, articles)
    ai_score, analysis = get_ai_score_with_context(text, top_articles)

    final_score = (ai_score * 0.7) + (sim_score * 0.3) if top_articles else ai_score

    return {
        "score": round(final_score, 1),
        "ai_score": round(ai_score, 1),
        "news_sim": round(sim_score, 1),
        "analysis": analysis,
        "matched_articles": top_articles,
        "has_articles": bool(top_articles)
    }

# =========================
# ルート
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    result = {
        "score": 0,
        "ai_score": 0,
        "news_sim": 0,
        "analysis": "",
        "matched_articles": [],
        "has_articles": False
    }

    input_text = ""

    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT text, final_score, created_at
        FROM judgments
        ORDER BY id DESC
        LIMIT 5
    """)
    history = cursor.fetchall()

    if request.method == "POST":
        input_text = request.form.get("text", "")

        if input_text.strip():
            result = final_judgement(input_text)

            cursor.execute(
                "INSERT INTO judgments (text, final_score, ai_score) VALUES (?, ?, ?)",
                (input_text, result["score"], result["ai_score"])
            )
            conn.commit()
        else:
            result["analysis"] = "文章を入力してください"

    conn.close()

    return render_template("index.html", result=result, history=history, input_text=input_text)

# =========================
# 起動（Render対応）
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
