from enum import Enum
from typing import Any, List, Literal, Optional, Union, Dict
import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class AgentState(str, Enum):
    """Agent execution states"""

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    ERROR = "ERROR"


class Function(BaseModel):
    name: str
    arguments: str


class MessageRole(str, Enum):
    """
    メッセージロールの定義
    """
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


class Message(BaseModel):
    """
    LLMとのやり取りに使用されるメッセージの基本スキーマ
    """
    role: str
    content: str
    
    @classmethod
    def system_message(cls, content: str) -> "Message":
        """
        システムメッセージを作成します
        """
        return cls(role=MessageRole.SYSTEM, content=content)
    
    @classmethod
    def user_message(cls, content: str) -> "Message":
        """
        ユーザーメッセージを作成します
        """
        return cls(role=MessageRole.USER, content=content)
    
    @classmethod
    def assistant_message(cls, content: str) -> "Message":
        """
        アシスタントメッセージを作成します
        """
        return cls(role=MessageRole.ASSISTANT, content=content)
    
    @classmethod
    def function_message(cls, content: str) -> "Message":
        """
        関数メッセージを作成します
        """
        return cls(role=MessageRole.FUNCTION, content=content)


class ToolParameter(BaseModel):
    """
    ツールパラメータの定義
    """
    name: str
    description: str
    required: bool = False
    type: str = "string"
    enum: Optional[List[str]] = None


class Tool(BaseModel):
    """
    ツールの定義
    """
    name: str
    description: str
    parameters: Dict[str, Union[ToolParameter, Dict[str, Any]]] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """
    ツール呼び出しの定義
    """
    id: str
    name: str
    arguments: Dict[str, Any]


class ChatResponse(BaseModel):
    """
    チャットレスポンスの標準形式
    """
    content: str
    tool_calls: Optional[List[ToolCall]] = None


class AIProvider(str, Enum):
    """
    サポートされているAIプロバイダー
    """
    OPENAI = "openai"
    GEMINI = "gemini"
    CLAUDE = "claude"
    GROQ_LLAMA = "groq_llama"
    GROQ_DEEPSEEK = "groq_deepseek"


class ModelIORequest(BaseModel):
    """
    統一されたモデル入力リクエスト形式
    """
    messages: List[Message]
    system_messages: Optional[List[Message]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[str] = "auto"
    temperature: Optional[float] = 0.0
    stream: bool = False
    provider: Optional[AIProvider] = None
    model: Optional[str] = None
    max_tokens: Optional[int] = None


class ModelIOResponse(BaseModel):
    """
    統一されたモデル出力レスポンス形式
    """
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    usage: Optional[Dict[str, int]] = None
    provider: AIProvider
    model: str


class Memory(BaseModel):
    messages: List[Message] = Field(default_factory=list)
    max_messages: int = Field(default=100)
    provider: Optional[AIProvider] = None

    def add_message(self, message: Message) -> None:
        """Add a message to memory"""
        self.messages.append(message)
        # Optional: Implement message limit
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def add_messages(self, messages: List[Message]) -> None:
        """Add multiple messages to memory"""
        self.messages.extend(messages)

    def clear(self) -> None:
        """Clear all messages"""
        self.messages.clear()

    def get_recent_messages(self, n: int) -> List[Message]:
        """Get n most recent messages"""
        return self.messages[-n:]

    def to_dict_list(self) -> List[dict]:
        """Convert messages to list of dicts"""
        return [msg.dict() for msg in self.messages]
    
    def check_context_limit(self) -> Dict:
        """
        現在のメッセージ履歴がコンテキストウィンドウ制限を超えているかチェック
        """
        from app.token_counter import TokenCounter
        
        provider_str = self.provider.value if self.provider else "openai"
        return TokenCounter.check_context_limit(self.messages, provider_str)


class ConversationThread(BaseModel):
    """
    会話スレッドを管理するためのクラス
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(default="New Conversation")
    memory: Memory = Field(default_factory=Memory)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    provider: Optional[AIProvider] = None
    
    def add_message(self, message: Message) -> None:
        """メッセージを追加"""
        self.memory.add_message(message)
        self.updated_at = datetime.now()
    
    def add_user_message(self, content: str) -> None:
        """ユーザーメッセージを追加"""
        self.add_message(Message.user_message(content))
    
    def add_assistant_message(self, content: str) -> None:
        """アシスタントメッセージを追加"""
        self.add_message(Message.assistant_message(content))
    
    def add_system_message(self, content: str) -> None:
        """システムメッセージを追加"""
        self.add_message(Message.system_message(content))
    
    def check_context_limit(self) -> Dict:
        """
        現在のスレッドがコンテキストウィンドウ制限を超えているかチェック
        """
        if self.provider and self.memory.provider != self.provider:
            self.memory.provider = self.provider
            
        return self.memory.check_context_limit()


class ConversationManager(BaseModel):
    """
    複数の会話スレッドを管理するためのクラス
    """
    threads: Dict[str, ConversationThread] = Field(default_factory=dict)
    current_thread_id: Optional[str] = None
    
    def create_thread(self, provider: Optional[AIProvider] = None) -> ConversationThread:
        """新しいスレッドを作成"""
        thread = ConversationThread(provider=provider)
        self.threads[thread.id] = thread
        self.current_thread_id = thread.id
        return thread
    
    def get_thread(self, thread_id: Optional[str] = None) -> Optional[ConversationThread]:
        """スレッドを取得"""
        if thread_id:
            return self.threads.get(thread_id)
        elif self.current_thread_id:
            return self.threads.get(self.current_thread_id)
        return None
    
    def get_current_thread(self) -> Optional[ConversationThread]:
        """現在のスレッドを取得"""
        return self.get_thread()
    
    def set_current_thread(self, thread_id: str) -> bool:
        """現在のスレッドを設定"""
        if thread_id in self.threads:
            self.current_thread_id = thread_id
            return True
        return False
    
    def delete_thread(self, thread_id: str) -> bool:
        """スレッドを削除"""
        if thread_id in self.threads:
            if self.current_thread_id == thread_id:
                self.current_thread_id = None
            del self.threads[thread_id]
            return True
        return False
