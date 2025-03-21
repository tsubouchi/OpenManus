import os
from typing import List, Optional, Union, Dict, Any
from app.schema import Message, ModelIORequest, ModelIOResponse, AIProvider, ToolCall
from app.llm_factory import LLMFactory
from app.logger import logger


class ModelIO:
    """
    異なるAIプロバイダー間で統一されたインターフェースを提供するクラス。
    リクエストとレスポンスを標準化し、適切なLLMインスタンスにルーティングします。
    """
    
    @staticmethod
    async def generate(request: ModelIORequest) -> ModelIOResponse:
        """
        統一されたリクエスト形式を使用してAIモデルにリクエストを送信し、
        標準化されたレスポンスを返します。
        """
        try:
            # リクエストからプロバイダーを決定
            provider = request.provider or AIProvider(os.getenv("AI_PROVIDER", "openai").lower())
            
            # 適切なLLMインスタンスを取得
            llm = LLMFactory.get_llm(provider)
            
            # ツールが指定されている場合はツール呼び出し用のメソッドを使用
            if request.tools:
                response = await llm.ask_tool(
                    messages=request.messages,
                    system_msgs=request.system_messages,
                    tools=request.tools,
                    tool_choice=request.tool_choice,
                    stream=request.stream,
                    temperature=request.temperature
                )
                
                # レスポンスを標準形式に変換
                tool_calls = [
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.arguments
                    ) for tc in response.tool_calls
                ] if hasattr(response, "tool_calls") and response.tool_calls else None
                
                return ModelIOResponse(
                    content=response.content,
                    tool_calls=tool_calls,
                    usage=getattr(response, "usage", None),
                    provider=provider,
                    model=os.getenv(f"{provider.upper()}_MODEL", "unknown")
                )
            else:
                # 通常のテキストリクエスト
                content = await llm.ask(
                    messages=request.messages,
                    system_msgs=request.system_messages,
                    stream=request.stream,
                    temperature=request.temperature
                )
                
                return ModelIOResponse(
                    content=content,
                    tool_calls=None,
                    usage=None,
                    provider=provider,
                    model=os.getenv(f"{provider.upper()}_MODEL", "unknown")
                )
                
        except Exception as e:
            logger.error(f"Error in ModelIO.generate: {str(e)}")
            raise
    
    @staticmethod
    async def quick_ask(query: str, system_prompt: Optional[str] = None) -> str:
        """
        簡単な質問に対して迅速に応答を取得するためのショートカットメソッド
        """
        messages = [Message.user_message(query)]
        system_messages = [Message.system_message(system_prompt)] if system_prompt else None
        
        request = ModelIORequest(
            messages=messages,
            system_messages=system_messages
        )
        
        response = await ModelIO.generate(request)
        return response.content 
        
    @staticmethod
    async def process_conversation_thread(
        thread: "ConversationThread",
        query: str,
        system_prompt: Optional[str] = None,
        provider: Optional[AIProvider] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
    ) -> str:
        """
        会話スレッドに対してリクエストを処理するメソッド
        会話履歴を保持しながらコンテキストウィンドウサイズを考慮します
        """
        try:
            # プロバイダーを決定
            provider = provider or AIProvider(os.getenv("AI_PROVIDER", "openai").lower())
            
            # 適切なLLMインスタンスを取得
            llm = LLMFactory.get_llm(provider)
            
            # システムメッセージを準備
            system_messages = [Message.system_message(system_prompt)] if system_prompt else None
            
            # LLMインスタンスを使用して会話スレッドを処理
            response = await llm.process_conversation_thread(
                thread=thread,
                user_message=query,
                system_msgs=system_messages,
                stream=stream,
                temperature=temperature
            )
            
            return response
            
        except Exception as e:
            error_message = f"Error in conversation thread processing: {str(e)}"
            logger.error(error_message)
            
            # エラーメッセージを会話スレッドに追加
            thread.add_assistant_message(error_message)
            
            return error_message 