import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()

# OpenAIクライアントの初期化
# 環境変数 OPENAI_API_KEY が設定されている必要があります
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CSV_FILE = "embeddings.csv"

def save_to_csv(text, embedding):
    """
    テキストとエンベディングをCSVファイルに保存する（簡易DB代わり）
    """
    # ベクトルを文字列として保存（カンマ区切りなど）
    # 大規模な場合はベクトルDBが推奨されますが、学習用途ではCSVやPandasが手軽です
    data = {
        "text": [text],
        "embedding": [str(embedding)]
    }
    df_new = pd.DataFrame(data)

    if os.path.exists(CSV_FILE):
        df_old = pd.read_csv(CSV_FILE)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(CSV_FILE, index=False)
    print(f"データを {CSV_FILE} に保存しました。")

def get_embedding(text, model="text-embedding-3-small"):
    """
    指定されたテキストの埋め込みベクトルを取得する
    """
    # 改行をスペースに置換（推奨される前処理）
    text = text.replace("\n", " ")

    # APIリクエスト
    response = client.embeddings.create(
        input=[text],
        model=model
    )

    # レスポンスからベクトルを抽出
    return response.data[0].embedding

if __name__ == "__main__":
    # サンプルテキストの一覧
    texts = [
        "こんにちは、世界！",
        "こんばんは、世界！",
        "サザエさんですよ！"
    ]

    for sample_text in texts:
        try:
            # 埋め込みベクトルの取得
            embedding = get_embedding(sample_text)

            print(f"テキスト: {sample_text}")
            print(f"ベクトルの次元数: {len(embedding)}")
            print(f"ベクトルの先頭5要素: {embedding[:5]}...")

            # CSV（DB代わり）に保存
            save_to_csv(sample_text, embedding)

        except Exception as e:
            print(f"エラーが発生しました ({sample_text}): {e}")

    # 保存された内容を表示してみる
    if os.path.exists(CSV_FILE):
        print("\n--- 現在の保存データ ---")
        print(pd.read_csv(CSV_FILE))
