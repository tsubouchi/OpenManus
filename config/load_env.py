import os
import dotenv
from pathlib import Path


def load_env_files():
    """
    環境変数ファイルを読み込みます。
    優先順位: .env.local > .env
    """
    # プロジェクトルートを特定
    project_root = Path(__file__).resolve().parent.parent.parent

    # .env.localがある場合はそれを優先
    env_local_path = project_root / ".env.local"
    env_path = project_root / ".env"

    # .env.localの読み込み
    if env_local_path.exists():
        dotenv.load_dotenv(env_local_path)
        print(f"Loaded environment variables from {env_local_path}")
    
    # .envの読み込み（.env.localで設定されていない変数のみ）
    if env_path.exists():
        dotenv.load_dotenv(env_path, override=False)
        print(f"Loaded additional environment variables from {env_path}")

    # 重要な環境変数が設定されているか確認
    provider = os.getenv("AI_PROVIDER", "openai").lower()
    
    if provider == "gemini":
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        if not gemini_key:
            print("WARNING: GEMINI_API_KEY is not set in environment variables")
    elif provider == "claude":
        claude_key = os.getenv("CLAUDE_API_KEY", "")
        if not claude_key:
            print("WARNING: CLAUDE_API_KEY is not set in environment variables")
    else:
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if not openai_key:
            print("WARNING: OPENAI_API_KEY is not set in environment variables")

    return {
        "provider": provider,
        "gemini_key": os.getenv("GEMINI_API_KEY", ""),
        "gemini_model": os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        "openai_key": os.getenv("OPENAI_API_KEY", ""),
        "openai_model": os.getenv("OPENAI_MODEL", "o3-mini-2025-01-31"),
        "claude_key": os.getenv("CLAUDE_API_KEY", ""),
        "claude_model": os.getenv("CLAUDE_MODEL", "claude-3-7-sonnet-20250219"),
        "server_port": int(os.getenv("SERVER_PORT", "3003")),
    }


# スクリプトとして実行された場合、環境変数を読み込んで表示
if __name__ == "__main__":
    env_vars = load_env_files()
    print("\nLoaded Environment Variables:")
    print(f"AI Provider: {env_vars['provider']}")
    
    if env_vars['provider'] == "gemini":
        print(f"Gemini Model: {env_vars['gemini_model']}")
        print(f"Gemini API Key: {'*' * 10}{env_vars['gemini_key'][-5:] if env_vars['gemini_key'] else 'NOT SET'}")
    elif env_vars['provider'] == "claude":
        print(f"Claude Model: {env_vars['claude_model']}")
        print(f"Claude API Key: {'*' * 10}{env_vars['claude_key'][-5:] if env_vars['claude_key'] else 'NOT SET'}")
    else:
        print(f"OpenAI Model: {env_vars['openai_model']}")
        print(f"OpenAI API Key: {'*' * 10}{env_vars['openai_key'][-5:] if env_vars['openai_key'] else 'NOT SET'}")
    
    print(f"Server Port: {env_vars['server_port']}") 