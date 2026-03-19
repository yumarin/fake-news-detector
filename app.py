from flask import Flask, render_template, request
import sqlite3
import os
import google.generativeai as genai
import feedparser

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ★ 日本語対応
from janome.tokenizer import Tokenizer

app = Flask(__name__)
t = Tokenizer()

# =========================
# RSS
# =========================
RSS_URLS = [
    "https://www3.nhk.or.jp/rss/news/cat0.xml",
    "https://news.yahoo.co.jp/rss/topics/top-picks.xml",
    "http://feeds.bbci.co.uk/news/rss.xml"
]

# =========================
# 日本語分かち書き
# =========================
def tokenize_japanese(text):
    return " ".join([token.surface for token in t.tokenize(text)])

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
# RSS取得
# =========================
def fetch_news():
    articles = []

    for url in RSS_URLS:
        feed = feedparser.parse(url)
        for entry in feed.entries[:10]:
            text = (entry.title + " " + entry.get("summary", "")).replace("\n", " ")
            articles.append({
                "title": entry.title,
                "link": entry.link,
                "text": text
            })

    return articles

# =========================
# 類似度計算（完全版）
# =========================
def calc_similarity(input_text, articles):
    if not articles:
        return 0, []

    # ★ 日本語分かち書き
    input_text = tokenize_japanese(input_text)
    texts = [tokenize_japanese(a["text"]) for a in articles]

    corpus = [input_text] + texts

    vectorizer = TfidfVectorizer(stop_words=["の", "に", "は", "を", "が", "と"])
    tfidf = vectorizer.fit_transform(corpus)

    sims = cosine_similarity(tfidf[0:1], tfidf[1:])[0]

    print("SIMS:", sims)

    max_sim = max(sims) if len(sims) > 0 else 0

    # ★ 閾値フィルター
    top_articles = []
    for i in sims.argsort()[::-1]:
        if sims[i] > 0.1:
            top_articles.append({
                "title": articles[i]["title"],
                "link": articles[i]["link"],
                "score": round(float(sims[i] * 100), 1)
            })

    return max_sim * 100, top_articles[:3]

# =========================
# AI判定（完全版）
# =========================
def get_ai_score_with_context(text, articles):
    if not GEMINI_API_KEY:
        return 0, "APIキー未設定"

    try:
        model = genai.GenerativeModel(MODEL_NAME)

        # =========================
        # 記事あり
        # =========================
        if articles:
            articles_text = "\n---\n".join([
                f"{a['title']} (一致度:{a['score']}%)"
                for a in articles
            ])

            prompt = f"""
以下の文章の信頼性を100点満点で評価してください。

【参考ニュース（関連性が高い順）】
{articles_text}

【評価対象】
{text}

以下の形式で必ず答えてください：

Score: 数値（0〜100）
Reason:
・ニュースとの一致度について
・不自然な点や怪しい点
・総合的な判断理由
"""

        # =========================
        # 記事なし
        # =========================
        else:
            prompt = f"""
以下の文章の信頼性を100点満点で評価してください。

※一致するニュースが見つかりませんでした。
一般知識と論理的観点のみで判断してください。

【評価対象】
{text}

以下の形式で必ず答えてください：

Score: 数値（0〜100）
Reason:
・論理的に正しいか
・不自然な点や誇張表現
・信頼できる情報かどうかの判断理由
"""

        response = model.generate_content(prompt)

        content = response.text

        score = 0
        reason = "解析失敗"

        for line in content.split("\n"):
            if "Score:" in line:
                try:
                    score = float(line.split(":")[1].strip())
                except:
                    pass

        if "Reason:" in content:
            reason = content.split("Reason:", 1)[1].strip()

        return score, reason

    except Exception as e:
        print("Gemini error:", e)
        return 0, str(e)

# =========================
# 総合判定（完成版）
# =========================
def final_judgement(text):
    articles = fetch_news()

    sim_score, top_articles = calc_similarity(text, articles)

    # ★ 記事なし対策
    if not top_articles:
        sim_score = 0

    ai_score, analysis = get_ai_score_with_context(text, top_articles)

    # ★ 重み切り替え
    if top_articles:
        final_score = (ai_score * 0.7) + (sim_score * 0.3)
    else:
        final_score = ai_score

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
