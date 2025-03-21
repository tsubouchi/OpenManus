from pydantic import Field
import os
from typing import Optional, Dict, List, Union

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
from app.schema import AIProvider, ConversationManager, ConversationThread
from app.model_io import ModelIO
from app.llm_factory import LLMFactory
from app.logger import logger


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

    def __init__(self, conversation_manager: Optional[ConversationManager] = None):
        super().__init__()
        self.state = AgentState.IDLE
        self.memory = Memory()
        self.module = AgentModule()
        
        # 会話スレッド管理
        self.conversation_manager = conversation_manager or ConversationManager()
        self.current_thread = self.conversation_manager.ensure_current_thread()
        
        # デフォルトのAIプロバイダー
        self.default_provider = AIProvider(os.getenv("AI_PROVIDER", "openai").lower())
        
        # スレッドのプロバイダーが設定されていない場合は設定
        if self.current_thread and not self.current_thread.provider:
            self.current_thread.provider = self.default_provider

    async def start(self):
        """エージェントを起動"""
        self.state = AgentState.RUNNING
        return "Agent started"

    async def stop(self):
        """エージェントを停止"""
        self.state = AgentState.IDLE
        return "Agent stopped"

    async def run(self, query: str) -> str:
        """
        ユーザーからのクエリを受け取り、現在のスレッドで処理
        """
        return await self.process_with_current_thread(query)

    async def process_with_current_thread(self, query: str) -> str:
        """
        現在の会話スレッドを使用してクエリを処理
        """
        try:
            # 現在のスレッドを取得（なければ作成）
            current_thread = self.conversation_manager.ensure_current_thread()
            if not current_thread.provider:
                current_thread.provider = self.default_provider
            
            # ユーザークエリをスレッドに追加
            current_thread.add_user_message(query)
            
            # コンテキスト制限をチェック
            context_info = current_thread.check_context_limit()
            if not context_info["is_within_limit"]:
                warning_message = (
                    f"⚠️ コンテキストウィンドウの制限（{context_info['max_tokens']}トークン）を超えています。"
                    f"現在のトークン数: {context_info['total_tokens']}トークン\n\n"
                    f"会話を新しいスレッドで開始することをお勧めします。"
                )
                current_thread.add_assistant_message(warning_message)
                return warning_message
            
            # システムプロンプト取得
            system_prompt = self.get_system_prompt()
            
            # AI応答を生成
            response = await ModelIO.process_conversation_thread(
                thread=current_thread,
                query=query,
                system_prompt=system_prompt,
                provider=current_thread.provider,
                stream=False,
                temperature=0.7
            )
            
            return response
        
        except Exception as e:
            error_message = f"エラーが発生しました: {str(e)}"
            logger.error(error_message)
            
            # エラーメッセージをスレッドに追加
            current_thread = self.conversation_manager.ensure_current_thread()
            current_thread.add_assistant_message(error_message)
            
            return error_message

    def create_new_thread(self, title: Optional[str] = None, provider: Optional[AIProvider] = None) -> str:
        """
        新しい会話スレッドを作成
        """
        # プロバイダーが指定されていない場合はデフォルトを使用
        provider = provider or self.default_provider
        
        # 新しいスレッドを作成
        new_thread = self.conversation_manager.create_thread(
            provider=provider,
            title=title or "New Conversation"
        )
        
        # 現在のスレッドを更新
        self.current_thread = new_thread
        
        return new_thread.id

    def get_thread_context_info(self, thread_id: Optional[str] = None) -> Dict:
        """
        スレッドのコンテキスト情報を取得
        """
        return self.conversation_manager.get_thread_context_info(thread_id)

    def get_all_threads_info(self) -> List[Dict[str, Union[str, int]]]:
        """
        すべてのスレッド情報を取得
        """
        return self.conversation_manager.list_threads()

    def switch_thread(self, thread_id: str) -> bool:
        """
        指定されたスレッドに切り替え
        """
        if self.conversation_manager.set_current_thread(thread_id):
            self.current_thread = self.conversation_manager.get_current_thread()
            return True
        return False

    def delete_thread(self, thread_id: str) -> bool:
        """
        指定されたスレッドを削除
        """
        result = self.conversation_manager.delete_thread(thread_id)
        if result and self.conversation_manager.current_thread_id is None:
            # 現在のスレッドが削除された場合は新しいスレッドを作成
            self.create_new_thread()
        return result

    def update_thread_title(self, thread_id: str, title: str) -> bool:
        """
        スレッドのタイトルを更新
        """
        return self.conversation_manager.rename_thread(thread_id, title)
