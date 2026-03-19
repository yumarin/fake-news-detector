from flask import Flask, render_template, request
import sqlite3
import os
import google.generativeai as genai

app = Flask(__name__)

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

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

    # 🔥 使えるモデル一覧をログに出す（超重要）
    print("=== AVAILABLE MODELS ===")
    try:
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                print(m.name)
    except Exception as e:
        print("Model list error:", e)

# =========================
# AI判定
# =========================
def get_ai_score(text):
    if not GEMINI_API_KEY:
        return 0, "APIキー未設定"

    try:
        # 🔥 安定モデル（まずこれ使え）
        model = genai.GenerativeModel("models/gemini-1.5-flash-002")

        response = model.generate_content(
            f"""
以下の文章の信頼性を100点満点で評価してください。
必ず以下の形式で答えてください：

Score: 数値
Reason: 理由

文章:
{text}
"""
        )

        if not response or not hasattr(response, "text"):
            return 0, "レスポンス取得失敗"

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
# ルート
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    result = {
        "score": 0,
        "ai_score": 0,
        "news_sim": 0,
        "analysis": ""
    }
    input_text = ""

    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("SELECT text, final_score, created_at FROM judgments ORDER BY id DESC LIMIT 5")
    history = cursor.fetchall()

    if request.method == "POST":
        input_text = request.form.get("text", "")

        ai_score, analysis = get_ai_score(input_text)

        result = {
            "score": ai_score,
            "analysis": analysis,
            "news_sim": 0,
            "ai_score": ai_score,
        }

        cursor.execute(
            "INSERT INTO judgments (text, final_score, ai_score) VALUES (?, ?, ?)",
            (input_text, ai_score, ai_score)
        )
        conn.commit()

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
