import os
import tiktoken
from typing import Dict, List, Union, Optional

from app.schema import Message
from app.logger import logger


class TokenCounter:
    """
    各種LLMモデルのトークン数をカウントするユーティリティクラス
    """
    
    # デフォルトエンコーディング
    _DEFAULT_ENCODING = "cl100k_base"  # OpenAIモデル用のデフォルトエンコーディング
    
    # 各プロバイダのエンコーディングマッピング
    _ENCODING_MAP = {
        "openai": "cl100k_base",       # OpenAI GPT系列
        "claude": "cl100k_base",       # Claude系列も同様のエンコーディングを使用
        "gemini": "cl100k_base",       # Geminiは正確には異なるが近似として使用
        "groq_llama": "cl100k_base",   # Llama系
        "groq_deepseek": "cl100k_base" # DeepSeek系
    }
    
    @classmethod
    def get_encoding(cls, provider: str) -> tiktoken.Encoding:
        """
        プロバイダに基づいてエンコーディングを取得
        """
        encoding_name = cls._ENCODING_MAP.get(provider, cls._DEFAULT_ENCODING)
        try:
            return tiktoken.get_encoding(encoding_name)
        except Exception as e:
            logger.error(f"エンコーディング取得エラー: {str(e)}")
            return tiktoken.get_encoding(cls._DEFAULT_ENCODING)
    
    @classmethod
    def count_tokens(cls, text: str, provider: str = "openai") -> int:
        """
        テキストのトークン数をカウント
        """
        if not text:
            return 0
            
        encoding = cls.get_encoding(provider)
        return len(encoding.encode(text))
    
    @classmethod
    def count_message_tokens(cls, 
                            messages: List[Union[Dict, Message]], 
                            provider: str = "openai") -> int:
        """
        メッセージリストのトークン数をカウント
        """
        token_count = 0
        encoding = cls.get_encoding(provider)
        
        # メッセージごとに処理
        for message in messages:
            # メッセージを共通形式に変換
            if isinstance(message, dict):
                role = message.get("role", "user")
                content = message.get("content", "")
            elif isinstance(message, Message):
                role = message.role
                content = message.content
            else:
                logger.warning(f"サポートされていないメッセージ型: {type(message)}")
                continue
                
            # ベースとなるトークン数を追加（ロールごとに追加される分）
            token_count += 4  # メッセージフォーマットのオーバーヘッド（概算）
            
            # コンテンツのトークン数を追加
            token_count += len(encoding.encode(content))
        
        # メッセージ全体に追加されるフォーマットトークン
        token_count += 2  # 終了トークンなど
        
        return token_count
    
    @classmethod
    def check_context_limit(cls, 
                           messages: List[Union[Dict, Message]], 
                           provider: str = "openai") -> Dict:
        """
        メッセージリストがコンテキストウィンドウ制限を超えているかチェック
        戻り値: {"is_within_limit": bool, "total_tokens": int, "max_tokens": int, "remaining_tokens": int}
        """
        # プロバイダに基づいて最大トークン数を取得
        max_tokens = cls.get_max_context_tokens(provider)
        
        # メッセージのトークン数を計算
        total_tokens = cls.count_message_tokens(messages, provider)
        
        # 残りのトークン数を計算
        remaining_tokens = max_tokens - total_tokens
        
        return {
            "is_within_limit": total_tokens <= max_tokens,
            "total_tokens": total_tokens,
            "max_tokens": max_tokens,
            "remaining_tokens": remaining_tokens
        }
    
    @staticmethod
    def get_max_context_tokens(provider: str) -> int:
        """
        プロバイダに基づいて最大コンテキストウィンドウトークン数を取得
        """
        # 環境変数から各プロバイダの最大トークン数を取得
        if provider == "openai":
            return int(os.getenv("OPENAI_MAX_CONTEXT_TOKENS", "200000"))
        elif provider == "claude":
            return int(os.getenv("CLAUDE_MAX_CONTEXT_TOKENS", "200000"))
        elif provider == "gemini":
            return int(os.getenv("GEMINI_MAX_CONTEXT_TOKENS", "1000000"))
        elif provider in ["groq_llama", "groq_deepseek"]:
            return int(os.getenv("GROQ_MAX_CONTEXT_TOKENS", "128000"))
        else:
            # デフォルト値
            return 16000
    
    @staticmethod
    def get_max_output_tokens(provider: str) -> int:
        """
        プロバイダに基づいて最大出力トークン数を取得
        """
        # 環境変数から各プロバイダの最大出力トークン数を取得
        if provider == "openai":
            return int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "4096"))
        elif provider == "claude":
            return int(os.getenv("CLAUDE_MAX_OUTPUT_TOKENS", "128000"))
        elif provider == "gemini":
            return int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "8192"))
        elif provider in ["groq_llama", "groq_deepseek"]:
            return int(os.getenv("GROQ_MAX_OUTPUT_TOKENS", "4096"))
        else:
            # デフォルト値
            return 4096 