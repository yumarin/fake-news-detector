from flask import Flask, render_template, request
import sqlite3
import os
import google.generativeai as genai
import feedparser

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# =========================
# RSS
# =========================
RSS_URLS = [
    "https://www3.nhk.or.jp/rss/news/cat0.xml",
    "https://news.yahoo.co.jp/rss/topics/top-picks.xml",
    "http://feeds.bbci.co.uk/news/rss.xml"
]

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

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# =========================
# RSS取得（リンク付き）
# =========================
def fetch_news():
    articles = []

    for url in RSS_URLS:
        feed = feedparser.parse(url)
        for entry in feed.entries[:10]:
            articles.append({
                "title": entry.title,
                "link": entry.link,
                "text": (entry.title + " " + entry.get("summary", "")).replace("\n", " ")
            })

    return articles

# =========================
# 類似度計算
# =========================
def calc_similarity(input_text, articles):
    if not articles:
        return 0, []

    corpus = [input_text] + [a["text"] for a in articles]

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(corpus)

    sims = cosine_similarity(tfidf[0:1], tfidf[1:])[0]

    max_sim = max(sims) if len(sims) > 0 else 0

    top_indices = sims.argsort()[-3:][::-1]
    top_articles = [articles[i] for i in top_indices]

    return max_sim * 100, top_articles

# =========================
# AI判定（記事込み）
# =========================
def get_ai_score_with_context(text, articles):
    if not GEMINI_API_KEY:
        return 0, "APIキー未設定"

    try:
        model = genai.GenerativeModel(MODEL_NAME)

        articles_text = "\n---\n".join([a["text"] for a in articles])

        response = model.generate_content(
            f"""
以下の文章の信頼性を100点満点で評価してください。

参考ニュース:
{articles_text}

評価対象:
{text}

必ず以下の形式で答えてください：
Score: 数値
Reason: 理由
"""
        )

        content = response.text

        score = 0
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
        return 0, str(e)

# =========================
# 総合判定
# =========================
def final_judgement(text):
    articles = fetch_news()
    sim_score, top_articles = calc_similarity(text, articles)

    ai_score, analysis = get_ai_score_with_context(text, top_articles)

    final_score = (ai_score * 0.7) + (sim_score * 0.3)

    return {
        "score": round(final_score, 1),
        "ai_score": ai_score,
        "news_sim": round(sim_score, 1),
        "analysis": analysis,
        "matched_articles": top_articles
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
        "matched_articles": []
    }

    input_text = ""

    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("SELECT text, final_score, created_at FROM judgments ORDER BY id DESC LIMIT 5")
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

    return render_template(
        "index.html",
        result=result,
        history=history,
        input_text=input_text
    )

# =========================
# 起動
# =========================
if __name__ == "__main__":
    app.run(debug=True)
