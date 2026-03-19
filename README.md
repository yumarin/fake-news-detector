# Fake News Detector β3.0

ニュースの信頼性をAIとデータ分析で判定するWebアプリです。

## 🔍 概要
入力された文章をもとに、
・最新ニュースとの類似度
・AIによる分析

を組み合わせて信頼度をスコア化します。

## ⚙️ 技術構成
- Python（Flask）
- TF-IDF（scikit-learn）
- Gemini API
- RSSフィード（NHK / Yahoo / BBC）
- SQLite（履歴保存）

## 🧠 判定ロジック
- ニュース類似度（30%）
- AI分析スコア（70%）

※一致するニュースがない場合はAIのみで評価

## 🚀 工夫した点
- 本文取得を避け、summaryを使用することで高速化
- 類似度＋AIのハイブリッド判定
- 上位記事のみを使用して精度と速度を両立

## 🌐 デモ
https://fake-news-detector-mq1e.onrender.com

## 🔧 今後の改善
- 上位記事の本文取得による精度向上
- UI/UXの改善
- 判定の視覚化（グラフなど）
