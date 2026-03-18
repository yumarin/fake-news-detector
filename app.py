import os
import sqlite3
from datetime import datetime
from flask import Flask, request, render_template, jsonify
import feedparser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import google.generativeai as genai

app = Flask(__name__)
DB_PATH = os.path.join(os.path.dirname(__file__), 'news_judge.db')

# Google Gemini APIの初期化
# 環境変数 GEMINI_API_KEY を設定してください。
# 無料で取得可能です: https://aistudio.google.com/app/apikey
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

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

def get_news(url):
    try:
        feed = feedparser.parse(url)
        return [entry.title for entry in feed.entries[:20]]
    except Exception as e:
        print(f"Error fetching news from {url}: {e}")
        return []

def calc_similarity(input_text, news_list):
    if not news_list:
        return 0.0
    texts = [input_text] + news_list
    vectorizer = TfidfVectorizer()
    try:
        tfidf = vectorizer.fit_transform(texts)
        sim = cosine_similarity(tfidf[0:1], tfidf[1:])
        return float(np.max(sim[0]))
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return 0.0

def get_ai_judgment(text):
    if not GEMINI_API_KEY:
        return 50, "AI解析（Gemini API）が設定されていません。APIキーを登録してください。"
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""
        あなたはフェイクニュース判定の専門家です。
        入力されたテキストの論理的な整合性、不自然な表現、事実関係の信憑性を分析し、100点満点中何点（高いほど信頼できる）で評価し、その理由を簡潔に述べてください。
        
        出力は必ず以下の形式にしてください：
        Score: [数字]
        Reason: [理由]
        
        テキスト：
        {text}
        """
        response = model.generate_content(prompt)
        content = response.text
        
        # スコアと理由を抽出
        lines = content.split('\n')
        score = 50
        reason = "AI解析に失敗しました。"
        for line in lines:
            if line.strip().startswith("Score:"):
                score_str = line.replace("Score:", "").strip()
                try:
                    score = float(score_str)
                except:
                    score = 50
            if line.strip().startswith("Reason:"):
                reason = line.replace("Reason:", "").strip()
        return score, reason
    except Exception as e:
        print(f"AI Judgment error: {e}")
        return 50, "AI解析中にエラーが発生しました。"

def calculate_final_score(news_sim, ai_score):
    news_base_score = min(100, news_sim * 250) 
    final_score = (news_base_score * 0.4) + (ai_score * 0.6)
    return round(final_score, 1)

def save_judgment(text, news_sim, ai_score, final_score, ai_analysis):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO judgments (text, news_sim_score, ai_score, final_score, ai_analysis)
        VALUES (?, ?, ?, ?, ?)
    ''', (text, news_sim, ai_score, final_score, ai_analysis))
    conn.commit()
    conn.close()

def get_history(limit=10):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT text, final_score, created_at FROM judgments ORDER BY created_at DESC LIMIT ?', (limit,))
    history = cursor.fetchall()
    conn.close()
    return history

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    input_text = ""
    if request.method == "POST":
        input_text = request.form.get("text", "")
        if input_text.strip():
            yahoo = get_news("https://news.yahoo.co.jp/rss/topics/top-picks.xml")
            google = get_news("https://news.google.com/rss?hl=ja&gl=JP&ceid=JP:ja")
            yahoo_sim = calc_similarity(input_text, yahoo)
            google_sim = calc_similarity(input_text, google)
            news_sim = max(yahoo_sim, google_sim)
            
            ai_score, ai_analysis = get_ai_judgment(input_text)
            final_score = calculate_final_score(news_sim, ai_score)
            save_judgment(input_text, news_sim, ai_score, final_score, ai_analysis)
            
            result = {
                "score": final_score,
                "ai_score": ai_score,
                "news_sim": round(news_sim, 3),
                "analysis": ai_analysis
            }
            
    history = get_history()
    return render_template("index.html", result=result, input_text=input_text, history=history)

@app.route("/api/judge", methods=["POST"])
def api_judge():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data['text']
    yahoo = get_news("https://news.yahoo.co.jp/rss/topics/top-picks.xml")
    google = get_news("https://news.google.com/rss?hl=ja&gl=JP&ceid=JP:ja")
    news_sim = max(calc_similarity(text, yahoo), calc_similarity(text, google))
    ai_score, ai_analysis = get_ai_judgment(text)
    final_score = calculate_final_score(news_sim, ai_score)
    
    save_judgment(text, news_sim, ai_score, final_score, ai_analysis)
    
    return jsonify({
        "status": "success",
        "score": final_score,
        "details": {
            "ai_score": ai_score,
            "news_similarity": news_sim,
            "ai_analysis": ai_analysis
        }
    })

if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=8080, debug=True)
