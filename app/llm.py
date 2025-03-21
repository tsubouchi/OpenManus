from typing import Dict, List, Literal, Optional, Union, Any
import os
import json
import aiohttp

from openai import (
    APIError,
    AsyncAzureOpenAI,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)
from tenacity import retry, stop_after_attempt, wait_random_exponential

from app.config import LLMSettings, config
from app.logger import logger  # Assuming a logger is set up in your app
from app.schema import Message, AIProvider, ConversationThread
from app.exceptions import ContextWindowExceededError


class LLM:
    """
    LLMの共通インタフェースを定義する基底クラス
    特定のLLMサービスごとに実装を行うためのテンプレート
    デフォルトではOpenAIのAPIを使用
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # 環境変数から設定を読み込み
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.api_url = os.getenv("OPENAI_API_URL", "https://api.openai.com")
        self.model = os.getenv("OPENAI_MODEL", "o3-mini-2025-01-31")
        self.endpoint = os.getenv("OPENAI_ENDPOINT", "v1/chat/completions")
        
        # コンテキストウィンドウサイズを取得
        self.max_context_tokens = int(os.getenv("OPENAI_MAX_CONTEXT_TOKENS", "200000"))
        self.max_output_tokens = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "4096"))
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY環境変数が設定されていません")
        
        self._initialized = True
    
    @staticmethod
    def format_messages(messages: List[Union[dict, Message]]) -> List[Dict]:
        """
        メッセージをLLM API形式に変換します
        """
        formatted_messages = []
        
        for message in messages:
            if isinstance(message, dict):
                formatted_messages.append({
                    "role": message.get("role", "user"),
                    "content": message.get("content", "")
                })
            elif isinstance(message, Message):
                formatted_messages.append({
                    "role": message.role,
                    "content": message.content
                })
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")
                
        return formatted_messages
        
    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
    )
    async def ask(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
    ) -> str:
        """
        LLM APIにリクエストを送信し、レスポンスを取得します
        """
        try:
            # メッセージの準備
            formatted_messages = []
            
            # システムメッセージが提供されている場合、最初に追加
            if system_msgs and len(system_msgs) > 0:
                for sys_msg in system_msgs:
                    if isinstance(sys_msg, dict):
                        formatted_messages.append({
                            "role": "system",
                            "content": sys_msg.get("content", "")
                        })
                    elif isinstance(sys_msg, Message):
                        if sys_msg.role == "system":
                            formatted_messages.append({
                                "role": "system",
                                "content": sys_msg.content
                            })
            
            # ユーザーおよびアシスタントメッセージを追加
            formatted_messages.extend(self.format_messages(messages))
            
            # トークン数をチェック
            self.check_token_limit(formatted_messages)
                
            # ペイロード構築
            payload = {
                "model": self.model,
                "messages": formatted_messages,
                "max_tokens": self.max_output_tokens,
                "temperature": 0.0 if temperature is None else temperature,
                "stream": stream
            }
            
            # リクエスト送信
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/{self.endpoint}",
                    json=payload,
                    headers={
                        "Content-Type": "application/json", 
                        "Authorization": f"Bearer {self.api_key}"
                    }
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"API error: {response.status} - {error_text}")
                        raise ValueError(f"API request failed with status {response.status}: {error_text}")
                    
                    if stream:
                        # ストリーミングレスポンスの処理
                        collected_content = []
                        async for line in response.content:
                            if line:
                                line_text = line.decode("utf-8").strip()
                                if line_text.startswith("data: "):
                                    data = line_text[6:]
                                    if data != "[DONE]":
                                        try:
                                            chunk = json.loads(data)
                                            delta = chunk["choices"][0]["delta"]
                                            if "content" in delta:
                                                content = delta["content"]
                                                collected_content.append(content)
                                                print(content, end="", flush=True)
                                        except json.JSONDecodeError:
                                            pass
                        
                        print()  # 最後に改行
                        return "".join(collected_content)
                    else:
                        # 通常レスポンスの処理
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
                    
        except Exception as e:
            logger.error(f"Error in API request: {str(e)}")
            raise
            
    async def ask_tool(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        tools: Optional[List[dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ):
        """
        ツール呼び出し対応のリクエストを実行します。
        """
        try:
            # メッセージの準備
            formatted_messages = []
            
            # システムメッセージが提供されている場合、最初に追加
            if system_msgs and len(system_msgs) > 0:
                for sys_msg in system_msgs:
                    if isinstance(sys_msg, dict):
                        formatted_messages.append({
                            "role": "system",
                            "content": sys_msg.get("content", "")
                        })
                    elif isinstance(sys_msg, Message):
                        if sys_msg.role == "system":
                            formatted_messages.append({
                                "role": "system",
                                "content": sys_msg.content
                            })
            
            # ユーザーおよびアシスタントメッセージを追加
            formatted_messages.extend(self.format_messages(messages))
            
            # トークン数をチェック
            self.check_token_limit(formatted_messages)
                
            # ペイロード構築
            payload = {
                "model": self.model,
                "messages": formatted_messages,
                "max_tokens": self.max_output_tokens,
                "temperature": 0.0
            }
            
            # ツール情報が提供されている場合、ペイロードに追加
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = tool_choice
            
            # リクエスト送信
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/{self.endpoint}",
                    json=payload,
                    headers={
                        "Content-Type": "application/json", 
                        "Authorization": f"Bearer {self.api_key}"
                    }
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"API error: {response.status} - {error_text}")
                        raise ValueError(f"API request failed with status {response.status}: {error_text}")
                    
                    # レスポンスの解析
                    result = await response.json()
                    response_message = result["choices"][0]["message"]
                    content = response_message.get("content", "")
                    tool_calls = response_message.get("tool_calls", [])
                    
                    # ツール呼び出し情報を含むレスポンスオブジェクトを返す
                    from types import SimpleNamespace
                    response = SimpleNamespace()
                    response.content = content
                    response.tool_calls = tool_calls
                    
                    return response
                    
        except Exception as e:
            logger.error(f"Error in tool API request: {str(e)}")
            raise
            
    def check_token_limit(self, messages: List[Dict]) -> bool:
        """
        メッセージのトークン数がモデルの制限を超えていないか確認します
        超えている場合はエラーを発生させます
        """
        from app.token_counter import TokenCounter
        
        # プロバイダ名を取得
        provider = "openai"
        
        # トークン数を計算
        result = TokenCounter.check_context_limit(messages, provider)
        
        if not result["is_within_limit"]:
            logger.warning(
                f"トークン数が制限を超えています: {result['total_tokens']} > {result['max_tokens']}"
            )
            raise ContextWindowExceededError(
                f"メッセージのトークン数がモデルの制限を超えています。会話を新しいスレッドで開始するか、古いメッセージを削除してください。",
                total_tokens=result['total_tokens'],
                max_tokens=result['max_tokens']
            )
        
        return True
        
    async def process_conversation_thread(
        self,
        thread: "ConversationThread",
        user_message: str,
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
    ) -> str:
        """
        会話スレッドに対してリクエストを処理します
        """
        # メッセージをスレッドに追加
        thread.add_user_message(user_message)
        
        # コンテキスト制限をチェック
        context_info = thread.check_context_limit()
        if not context_info["is_within_limit"]:
            # コンテキスト制限を超えている場合、制限に関する情報をレスポンスとしてアシスタントメッセージを追加
            warning_message = (
                f"⚠️ コンテキストウィンドウの制限（{context_info['max_tokens']}トークン）を超えています。"
                f"現在のトークン数: {context_info['total_tokens']}トークン\n\n"
                f"新しい会話スレッドを開始することをお勧めします。"
            )
            thread.add_assistant_message(warning_message)
            return warning_message
        
        try:
            # AIモデルに問い合わせ
            response = await self.ask(
                messages=thread.memory.messages,
                system_msgs=system_msgs,
                stream=stream,
                temperature=temperature
            )
            
            # レスポンスをスレッドに追加
            thread.add_assistant_message(response)
            
            return response
        except Exception as e:
            error_message = f"エラーが発生しました: {str(e)}"
            logger.error(error_message)
            thread.add_assistant_message(error_message)
            return error_message
