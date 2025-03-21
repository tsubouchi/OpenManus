from pydantic import Field
import os
from typing import Optional, Dict, List

from app.agent.toolcall import ToolCallAgent
from app.prompt.manus import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.tool import Terminate, ToolCollection
from app.tool.browser_use_tool import BrowserUseTool
from app.tool.file_saver import FileSaver
from app.tool.google_search import GoogleSearch
from app.tool.python_execute import PythonExecute
from app.agent.agent_state import AgentState
from app.agent.memory import Memory
from app.agent.agent_module import AgentModule
from app.agent.conversation_manager import ConversationManager
from app.agent.ai_provider import AIProvider
from app.agent.llm_factory import LLMFactory


class Manus(ToolCallAgent):
    """
    A versatile general-purpose agent that uses planning to solve various tasks.

    This agent extends PlanningAgent with a comprehensive set of tools and capabilities,
    including Python execution, web browsing, file operations, and information retrieval
    to handle a wide range of user requests.
    """

    name: str = "Manus"
    description: str = (
        "A versatile agent that can solve various tasks using multiple tools"
    )

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    # Add general-purpose tools to the tool collection
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(), GoogleSearch(), BrowserUseTool(), FileSaver(), Terminate()
        )
    )

    def __init__(self):
        super().__init__()
        self.state = AgentState.IDLE
        self.memory = Memory()
        self.module = AgentModule()
        self.conversation_manager = ConversationManager()
        self.current_thread = self.conversation_manager.create_thread()

    async def start(self):
        """Start the agent"""
        logger.info("Starting Manus agent...")
        self.state = AgentState.RUNNING

    async def stop(self):
        """Stop the agent"""
        logger.info("Stopping Manus agent...")
        self.state = AgentState.IDLE

    async def ask(self, query: str) -> str:
        """
        Ask the agent a question and get a response
        """
        raise NotImplementedError("Subclasses must implement this method")

    async def process_with_current_thread(self, query: str) -> str:
        """
        現在のスレッドを使用してクエリを処理します
        """
        if not self.current_thread:
            self.current_thread = self.conversation_manager.create_thread()
            
        # AIプロバイダーの取得
        provider = os.getenv("AI_PROVIDER", "openai")
        ai_provider = None
        try:
            ai_provider = AIProvider(provider.lower())
        except ValueError:
            logger.warning(f"Invalid AI provider: {provider}, using default")
            ai_provider = AIProvider.OPENAI
            
        # プロバイダーをスレッドに設定
        self.current_thread.provider = ai_provider
            
        # LLMファクトリーからLLMインスタンスを取得
        llm = LLMFactory.get_llm()
        
        # スレッドを使用してクエリを処理
        response = await llm.process_conversation_thread(
            thread=self.current_thread,
            user_message=query
        )
        
        return response
        
    def create_new_thread(self) -> str:
        """
        新しい会話スレッドを作成します
        """
        # AIプロバイダーの取得
        provider = os.getenv("AI_PROVIDER", "openai")
        ai_provider = None
        try:
            ai_provider = AIProvider(provider.lower())
        except ValueError:
            logger.warning(f"Invalid AI provider: {provider}, using default")
            ai_provider = AIProvider.OPENAI
            
        # 新しいスレッドを作成
        self.current_thread = self.conversation_manager.create_thread(provider=ai_provider)
        
        return self.current_thread.id
        
    def get_thread_context_info(self, thread_id: Optional[str] = None) -> Dict:
        """
        指定されたスレッド（またはデフォルトで現在のスレッド）のコンテキスト情報を取得します
        """
        thread = None
        if thread_id:
            thread = self.conversation_manager.get_thread(thread_id)
        else:
            thread = self.current_thread
            
        if not thread:
            return {
                "error": "Thread not found",
                "thread_id": thread_id
            }
            
        # コンテキスト制限情報を取得
        context_info = thread.check_context_limit()
        
        # スレッド情報を追加
        return {
            "thread_id": thread.id,
            "title": thread.title,
            "message_count": len(thread.memory.messages),
            "created_at": thread.created_at.isoformat(),
            "updated_at": thread.updated_at.isoformat(),
            "provider": thread.provider.value if thread.provider else None,
            "is_within_limit": context_info["is_within_limit"],
            "total_tokens": context_info["total_tokens"],
            "max_tokens": context_info["max_tokens"],
            "remaining_tokens": context_info["remaining_tokens"],
            "token_usage_percent": round(context_info["total_tokens"] / context_info["max_tokens"] * 100, 2)
        }
        
    def get_all_threads_info(self) -> List[Dict]:
        """
        すべてのスレッドの基本情報を取得します
        """
        result = []
        for thread_id, thread in self.conversation_manager.threads.items():
            context_info = thread.check_context_limit()
            result.append({
                "thread_id": thread.id,
                "title": thread.title,
                "message_count": len(thread.memory.messages),
                "created_at": thread.created_at.isoformat(),
                "updated_at": thread.updated_at.isoformat(),
                "provider": thread.provider.value if thread.provider else None,
                "is_current": thread_id == self.conversation_manager.current_thread_id,
                "token_usage_percent": round(context_info["total_tokens"] / context_info["max_tokens"] * 100, 2)
            })
        return result
        
    def switch_thread(self, thread_id: str) -> bool:
        """
        指定されたスレッドに切り替えます
        """
        if self.conversation_manager.set_current_thread(thread_id):
            self.current_thread = self.conversation_manager.get_thread(thread_id)
            return True
        return False
        
    def delete_thread(self, thread_id: str) -> bool:
        """
        指定されたスレッドを削除します
        """
        return self.conversation_manager.delete_thread(thread_id)
        
    def update_thread_title(self, thread_id: str, title: str) -> bool:
        """
        指定されたスレッドのタイトルを更新します
        """
        thread = self.conversation_manager.get_thread(thread_id)
        if thread:
            thread.title = title
            return True
        return False
