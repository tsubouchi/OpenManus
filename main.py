import asyncio
import os

from app.agent.manus import Manus
from app.logger import logger
from config.load_env import load_env_files


async def main():
    # 環境変数を読み込む
    env_vars = load_env_files()
    
    # 使用するAIモデルの情報をログに出力
    provider = env_vars["provider"]
    logger.info(f"Using AI provider: {provider.upper()}")
    
    if provider == "gemini":
        logger.info(f"Gemini model: {env_vars['gemini_model']}")
    else:
        logger.info(f"OpenAI model: {env_vars['openai_model']}")
    
    # エージェントの初期化
    agent = Manus()
    
    while True:
        try:
            prompt = input("Enter your prompt (or 'exit' to quit): ")
            if prompt.lower() == "exit":
                logger.info("Goodbye!")
                break
            if prompt.strip().isspace():
                logger.warning("Skipping empty prompt.")
                continue
            logger.warning("Processing your request...")
            await agent.run(prompt)
        except KeyboardInterrupt:
            logger.warning("Goodbye!")
            break


if __name__ == "__main__":
    asyncio.run(main())
