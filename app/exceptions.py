class ToolError(Exception):
    """Raised when a tool encounters an error."""

    def __init__(self, message):
        self.message = message


class ContextWindowExceededError(Exception):
    """コンテキストウィンドウの制限を超えた場合に発生する例外"""

    def __init__(self, message, total_tokens=None, max_tokens=None):
        self.message = message
        self.total_tokens = total_tokens
        self.max_tokens = max_tokens
        
    def __str__(self):
        return f"{self.message} (使用トークン: {self.total_tokens}, 最大トークン: {self.max_tokens})"


class TokenCountingError(Exception):
    """トークン数の計算中にエラーが発生した場合の例外"""

    def __init__(self, message, provider=None):
        self.message = message
        self.provider = provider
        
    def __str__(self):
        if self.provider:
            return f"{self.message} (プロバイダ: {self.provider})"
        return self.message
