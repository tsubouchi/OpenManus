import os
import json
from typing import Dict, List, Optional, Union
from anthropic import AsyncAnthropic

from app.schema import Message, Function, ConversationThread
from app.logger import logger


class ClaudeLLM:
    """
    Anthropic ClaudeのLLMインターフェース実装クラス
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
        self.api_key = os.getenv("CLAUDE_API_KEY", "")
        
        if not self.api_key:
            raise ValueError("CLAUDE_API_KEY環境変数が設定されていません")
            
        self.model = os.getenv("CLAUDE_MODEL", "claude-3-7-sonnet-20250219")
        
        # コンテキストウィンドウサイズを取得
        self.max_context_tokens = int(os.getenv("CLAUDE_MAX_CONTEXT_TOKENS", "200000"))
        self.max_output_tokens = int(os.getenv("CLAUDE_MAX_OUTPUT_TOKENS", "128000"))
        
        # Anthropic API Client初期化
        self.client = AsyncAnthropic(
            api_key=self.api_key,
        )
        
        self._initialized = True
        
    @staticmethod
    def format_messages(messages: List[Union[dict, Message]]) -> List[Dict]:
        """
        メッセージをClaude形式に変換する
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
        
    async def ask(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Claude APIに問い合わせ、レスポンスを取得する
        """
        try:
            # メッセージの準備
            formatted_messages = []
            
            # システムメッセージが提供されている場合、別途保持
            system_message = ""
            if system_msgs and len(system_msgs) > 0:
                system_contents = []
                for sys_msg in system_msgs:
                    if isinstance(sys_msg, dict):
                        system_contents.append(sys_msg.get("content", ""))
                    elif isinstance(sys_msg, Message):
                        if sys_msg.role == "system":
                            system_contents.append(sys_msg.content)
                system_message = "\n".join(system_contents)
            
            # ユーザーおよびアシスタントメッセージを追加
            formatted_messages.extend(self.format_messages(messages))
            
            # トークン数をチェック
            self.check_token_limit(formatted_messages)
            
            # Claude API形式に変換
            claude_messages = []
            
            for msg in formatted_messages:
                role = msg["role"]
                content = msg["content"]
                
                if role == "user":
                    claude_messages.append({"role": "user", "content": content})
                elif role == "assistant":
                    claude_messages.append({"role": "assistant", "content": content})
                elif role == "system":
                    # システムメッセージが複数ある場合、最後のものを使用
                    system_message = content
            
            # レスポンスを取得
            if stream:
                # ストリーミングレスポンスの処理
                response_stream = await self.client.messages.create(
                    model=self.model,
                    messages=claude_messages,
                    system=system_message if system_message else None,
                    temperature=0.6 if temperature is None else temperature,
                    max_tokens=self.max_output_tokens,
                    stream=True
                )
                
                collected_content = []
                async for chunk in response_stream:
                    if hasattr(chunk, "delta") and chunk.delta.text:
                        content = chunk.delta.text
                        collected_content.append(content)
                        print(content, end="", flush=True)
                
                print()  # 最後に改行
                return "".join(collected_content)
            else:
                # 通常レスポンスの処理
                response = await self.client.messages.create(
                    model=self.model,
                    messages=claude_messages,
                    system=system_message if system_message else None,
                    temperature=0.6 if temperature is None else temperature,
                    max_tokens=self.max_output_tokens
                )
                
                return response.content[0].text
                
        except Exception as e:
            logger.error(f"Error in Claude API request: {str(e)}")
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
            
            # システムメッセージが提供されている場合、別途保持
            system_message = ""
            if system_msgs and len(system_msgs) > 0:
                system_contents = []
                for sys_msg in system_msgs:
                    if isinstance(sys_msg, dict):
                        system_contents.append(sys_msg.get("content", ""))
                    elif isinstance(sys_msg, Message):
                        if sys_msg.role == "system":
                            system_contents.append(sys_msg.content)
                system_message = "\n".join(system_contents)
            
            # ユーザーおよびアシスタントメッセージを追加
            formatted_messages.extend(self.format_messages(messages))
            
            # トークン数をチェック
            self.check_token_limit(formatted_messages)
            
            # Claude API形式に変換
            claude_messages = []
            
            for msg in formatted_messages:
                role = msg["role"]
                content = msg["content"]
                
                if role == "user":
                    claude_messages.append({"role": "user", "content": content})
                elif role == "assistant":
                    claude_messages.append({"role": "assistant", "content": content})
                elif role == "system":
                    # システムメッセージが複数ある場合、最後のものを使用
                    system_message = content
            
            # ツールをClaude形式に変換
            claude_tools = []
            
            if tools:
                for tool in tools:
                    if "function" in tool:
                        function_data = tool["function"]
                        claude_tools.append({
                            "name": function_data.get("name", ""),
                            "description": function_data.get("description", ""),
                            "input_schema": function_data.get("parameters", {})
                        })
            
            # レスポンス取得
            response = await self.client.messages.create(
                model=self.model,
                messages=claude_messages,
                system=system_message if system_message else None,
                temperature=0.6,
                max_tokens=self.max_output_tokens,
                tools=claude_tools if claude_tools else None
            )
            
            # レスポンスからツール呼び出し情報を取得
            content = response.content[0].text if response.content and response.content[0].text else ""
            
            # Tool callsを検出する
            tool_calls = []
            
            if hasattr(response, "content"):
                for content_block in response.content:
                    if hasattr(content_block, "type") and content_block.type == "tool_use":
                        tool_use = content_block.tool_use
                        tool_calls.append({
                            "id": f"call_{len(tool_calls)}",
                            "type": "function",
                            "function": {
                                "name": tool_use.name,
                                "arguments": json.dumps(tool_use.input)
                            }
                        })
            
            # SimpleNamespaceを使用してレスポンスオブジェクトを作成
            from types import SimpleNamespace
            result = SimpleNamespace()
            result.content = content
            result.tool_calls = tool_calls
            
            return result
                
        except Exception as e:
            logger.error(f"Error in Claude API tool request: {str(e)}")
            raise
            
    def check_token_limit(self, messages: List[Union[dict, Message]]) -> bool:
        """
        メッセージのトークン数がモデルの制限を超えていないか確認します
        超えている場合はエラーを発生させます
        """
        from app.token_counter import TokenCounter
        
        # プロバイダ名を取得
        provider = "claude"
        
        # トークン数を計算
        result = TokenCounter.check_context_limit(messages, provider)
        
        if not result["is_within_limit"]:
            logger.warning(
                f"トークン数が制限を超えています: {result['total_tokens']} > {result['max_tokens']}"
            )
            raise ValueError(
                f"メッセージのトークン数 ({result['total_tokens']}) がClaudeモデルの制限 ({result['max_tokens']}) を超えています。"
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