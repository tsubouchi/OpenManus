import os
import google.generativeai as genai
from typing import Any, Dict, List, Optional, Union

from app.schema import Message, Function
from app.logger import logger


class GeminiLLM:
    """
    Google GeminiモデルのLLMインターフェース実装クラス
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
            
        # 環境変数からGoogle API Keyを取得
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY環境変数が設定されていません")
            
        # モデル情報を取得
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-flash")
        
        # コンテキストウィンドウサイズを取得
        self.max_context_tokens = int(os.getenv("GEMINI_MAX_CONTEXT_TOKENS", "1000000"))
        self.max_output_tokens = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "8192"))
        
        # Gemini APIの初期化
        genai.configure(api_key=self.api_key)
        
        # モデルの生成設定
        self.generation_config = {
            "temperature": 0.6,
            "top_p": 0.95,
            "max_output_tokens": self.max_output_tokens,
        }
        
        self._initialized = True
        
    @staticmethod
    def format_messages(messages: List[Union[dict, Message]]) -> List[Dict]:
        """
        メッセージをGemini形式に変換する
        """
        formatted_messages = []
        
        for message in messages:
            if isinstance(message, dict):
                role = message.get("role", "user")
                content = message.get("content", "")
                
                # Geminiでは "user", "model", "system" のロールが有効
                if role == "assistant":
                    role = "model"
                
                formatted_messages.append({
                    "role": role,
                    "parts": [{"text": content}],
                })
            elif isinstance(message, Message):
                role = message.role
                content = message.content
                
                # Geminiでは "user", "model", "system" のロールが有効
                if role == "assistant":
                    role = "model"
                
                formatted_messages.append({
                    "role": role,
                    "parts": [{"text": content}],
                })
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")
                
        return formatted_messages
        
    async def ask(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Gemini APIに問い合わせ、レスポンスを取得する
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
                            "parts": [{"text": sys_msg.get("content", "")}],
                        })
                    elif isinstance(sys_msg, Message):
                        if sys_msg.role == "system":
                            formatted_messages.append({
                                "role": "system",
                                "parts": [{"text": sys_msg.content}],
                            })
            
            # ユーザーおよびアシスタントメッセージを追加
            formatted_messages.extend(self.format_messages(messages))
            
            # トークン数をチェック
            self.check_token_limit(formatted_messages)
            
            # モデルを初期化
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    **self.generation_config,
                    **({"temperature": temperature} if temperature is not None else {})
                }
            )
            
            # チャットセッションを作成してメッセージを送信
            chat = model.start_chat(history=formatted_messages)
            last_user_message = ""
            
            # 最後のユーザーメッセージを取得
            for msg in reversed(formatted_messages):
                if msg["role"] == "user":
                    if "parts" in msg and msg["parts"] and "text" in msg["parts"][0]:
                        last_user_message = msg["parts"][0]["text"]
                        break
            
            if not last_user_message:
                raise ValueError("ユーザーメッセージが見つかりません")
            
            # レスポンスを取得
            if stream:
                # ストリーミングレスポンスの処理
                response_stream = chat.send_message(last_user_message, stream=True)
                collected_content = []
                
                for chunk in response_stream:
                    if hasattr(chunk, "text"):
                        text = chunk.text
                        collected_content.append(text)
                        print(text, end="", flush=True)
                
                print()  # 最後に改行
                return "".join(collected_content)
            else:
                # 通常レスポンスの処理
                response = chat.send_message(last_user_message)
                return response.text
                
        except Exception as e:
            logger.error(f"Error in Gemini API request: {str(e)}")
            raise

    async def ask_tool(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ):
        """
        ツール呼び出し対応のリクエストを実行します
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
                            "parts": [{"text": sys_msg.get("content", "")}],
                        })
                    elif isinstance(sys_msg, Message):
                        if sys_msg.role == "system":
                            formatted_messages.append({
                                "role": "system",
                                "parts": [{"text": sys_msg.content}],
                            })
            
            # ユーザーおよびアシスタントメッセージを追加
            formatted_messages.extend(self.format_messages(messages))
            
            # トークン数をチェック
            self.check_token_limit(formatted_messages)
            
            # ツールをGemini形式に変換
            function_declarations = []
            
            if tools:
                for tool in tools:
                    if "function" in tool:
                        function_data = tool["function"]
                        # paramsが存在するかどうか確認
                        parameters = {}
                        if "parameters" in function_data:
                            parameters = function_data["parameters"]
                        
                        function_declarations.append({
                            "name": function_data.get("name", ""),
                            "description": function_data.get("description", ""),
                            "parameters": parameters
                        })
            
            # モデルを初期化（ツール呼び出し対応）
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config,
                tools=function_declarations if function_declarations else None
            )
            
            # チャットセッションを作成してメッセージを送信
            chat = model.start_chat(history=formatted_messages)
            last_user_message = ""
            
            # 最後のユーザーメッセージを取得
            for msg in reversed(formatted_messages):
                if msg["role"] == "user":
                    if "parts" in msg and msg["parts"] and "text" in msg["parts"][0]:
                        last_user_message = msg["parts"][0]["text"]
                        break
            
            if not last_user_message:
                raise ValueError("ユーザーメッセージが見つかりません")
            
            # レスポンスを取得
            response = chat.send_message(last_user_message)
            
            # レスポンスからツール呼び出し情報を取得
            from types import SimpleNamespace
            result = SimpleNamespace()
            
            # テキスト内容の取得
            result.content = response.text
            
            # ツール呼び出しの取得
            result.tool_calls = []
            
            if hasattr(response, "candidates") and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, "content") and candidate.content:
                        content = candidate.content
                        if hasattr(content, "parts") and content.parts:
                            for part in content.parts:
                                if hasattr(part, "function_call") and part.function_call:
                                    function_call = part.function_call
                                    result.tool_calls.append({
                                        "id": f"call_{len(result.tool_calls)}",
                                        "type": "function",
                                        "function": {
                                            "name": function_call.name,
                                            "arguments": function_call.args
                                        }
                                    })
            
            return result
                
        except Exception as e:
            logger.error(f"Error in Gemini API tool request: {str(e)}")
            raise
            
    def check_token_limit(self, messages) -> bool:
        """
        メッセージのトークン数がモデルの制限を超えていないか確認します
        超えている場合はエラーを発生させます
        """
        from app.token_counter import TokenCounter
        
        # プロバイダ名を取得
        provider = "gemini"
        
        # トークン数を計算
        # Gemini形式のメッセージを標準形式に変換する
        standardized_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "model":
                role = "assistant"  # モデルのロールをアシスタントに戻す
            
            content = ""
            if "parts" in msg and msg["parts"] and "text" in msg["parts"][0]:
                content = msg["parts"][0]["text"]
                
            standardized_messages.append({
                "role": role,
                "content": content
            })
        
        result = TokenCounter.check_context_limit(standardized_messages, provider)
        
        if not result["is_within_limit"]:
            logger.warning(
                f"トークン数が制限を超えています: {result['total_tokens']} > {result['max_tokens']}"
            )
            raise ValueError(
                f"メッセージのトークン数 ({result['total_tokens']}) がGeminiモデルの制限 ({result['max_tokens']}) を超えています。"
                f"会話を新しいスレッドで開始するか、古いメッセージを削除してください。"
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