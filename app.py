from flask import Flask, render_template, request
import sqlite3

app = Flask(__name__)

def init_db():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS judgments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT,
        final_score REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

init_db()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    input_text = ""

    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    # 履歴取得
    cursor.execute("SELECT text, final_score, created_at FROM judgments ORDER BY id DESC LIMIT 5")
    history = cursor.fetchall()

    if request.method == "POST":
        input_text = request.form["text"]

        # ←ここは絶対壊れない固定値
        result = {
            "score": 50,
            "ai_score": 50,
            "news_sim": 0.5,
            "analysis": "テスト分析（正常動作）"
        }

        cursor.execute(
            "INSERT INTO judgments (text, final_score) VALUES (?, ?)",
            (input_text, result["score"])
        )
        conn.commit()

    conn.close()

    return render_template(
        "index.html",
        result=result,
        history=history,
        input_text=input_text
    )

if __name__ == "__main__":
    app.run(debug=True)
