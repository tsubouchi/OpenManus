import os
from typing import Dict, List, Optional, Union, Any

from app.logger import logger
from app.schema import Message, AIProvider, ConversationThread
from app.llm_gemini import GeminiLLM
from app.llm_claude import ClaudeLLM
from app.llm_openai import OpenAILLM
from app.llm_llama_groq import LlamaGroqLLM
from app.llm_deepseek_groq import DeepSeekGroqLLM


class LLMFactory:
    """
    LLMのファクトリークラス
    環境変数に基づいて適切なLLMインスタンスを提供します
    """
    
    @staticmethod
    def get_llm(provider: Optional[str] = None) -> Union[GeminiLLM, OpenAILLM, ClaudeLLM, LlamaGroqLLM, DeepSeekGroqLLM]:
        """
        指定されたプロバイダに基づいてLLMインスタンスを取得します
        未指定の場合は環境変数から読み込みます
        """
        if not provider:
            # 環境変数からプロバイダを取得
            provider = os.getenv("AI_PROVIDER", "gemini").lower()
        
        # プロバイダに基づいてLLMを初期化して返す
        if provider == AIProvider.GEMINI.value.lower() or provider == "gemini":
            logger.info("GeminiLLMを使用します")
            return GeminiLLM()
        elif provider == AIProvider.OPENAI.value.lower() or provider == "openai":
            logger.info("OpenAILLMを使用します")
            return OpenAILLM()
        elif provider == AIProvider.CLAUDE.value.lower() or provider == "claude":
            logger.info("ClaudeLLMを使用します")
            return ClaudeLLM()
        elif provider == AIProvider.GROQ_LLAMA.value.lower() or provider == "groq_llama":
            logger.info("LlamaGroqLLMを使用します")
            return LlamaGroqLLM()
        elif provider == AIProvider.GROQ_DEEPSEEK.value.lower() or provider == "groq_deepseek":
            logger.info("DeepSeekGroqLLMを使用します")
            return DeepSeekGroqLLM()
        else:
            # デフォルトはGemini
            logger.warning(f"不明なプロバイダ '{provider}' が指定されました。デフォルトのGeminiLLMを使用します。")
            return GeminiLLM()
    
    @classmethod
    async def ask_with_provider(
        cls,
        messages: List[Union[dict, Message]],
        provider: Optional[str] = None,
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
    ) -> str:
        """
        指定されたプロバイダを使用してAIに問い合わせます
        """
        llm = cls.get_llm(provider)
        return await llm.ask(
            messages=messages,
            system_msgs=system_msgs,
            stream=stream,
            temperature=temperature
        )
    
    @classmethod
    async def ask_tool_with_provider(
        cls,
        messages: List[Union[dict, Message]],
        provider: Optional[str] = None,
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ) -> Any:
        """
        指定されたプロバイダを使用してツール呼び出し対応のリクエストを実行します
        """
        llm = cls.get_llm(provider)
        return await llm.ask_tool(
            messages=messages,
            system_msgs=system_msgs,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs
        )
        
    @classmethod
    async def process_conversation_thread(
        cls,
        thread: "ConversationThread",
        user_message: str,
        provider: Optional[str] = None,
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
    ) -> str:
        """
        指定されたプロバイダを使用して会話スレッドの処理を行います
        """
        llm = cls.get_llm(provider)
        return await llm.process_conversation_thread(
            thread=thread,
            user_message=user_message,
            system_msgs=system_msgs,
            stream=stream,
            temperature=temperature
        )
        
    @classmethod
    def get_context_window_size(cls, provider: Optional[str] = None) -> int:
        """
        指定されたプロバイダのコンテキストウィンドウサイズを取得します
        """
        llm = cls.get_llm(provider)
        return llm.max_context_tokens if hasattr(llm, 'max_context_tokens') else 4096 