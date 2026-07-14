"""
配置向导 — 交互式创建/更新 .env 文件

用法:
    python setup.py          # 交互式向导
    python setup.py --quick  # 快速模式：只创建默认 .env（不填 API Key）

这个脚本会：
1. 检测是否已有 .env 文件
2. 引导用户选择 LLM Provider
3. 引导用户填入 API Key
4. 生成或更新 .env 文件
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


# ============================================================================
# 常量
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"

PROVIDERS = {
    "1": {"key": "openai", "name": "OpenAI (GPT-4o-mini)", "env_key": "OPENAI_API_KEY"},
    "2": {"key": "anthropic", "name": "Anthropic (Claude)", "env_key": "ANTHROPIC_API_KEY"},
    "3": {"key": "groq", "name": "Groq (Qwen)", "env_key": "GROQ_API_KEY"},
}


def print_banner():
    print()
    print("=" * 60)
    print("  🎓 AI English Tutor — 配置向导")
    print("=" * 60)
    print()


def check_existing_env() -> bool:
    """检查是否已有 .env 文件"""
    if ENV_FILE.exists():
        print(f"📄 检测到现有配置文件: {ENV_FILE}")
        print()
        answer = input("  是否覆盖现有配置？(y/N): ").strip().lower()
        if answer != "y":
            print("  已取消配置。如需重新配置，请手动编辑 .env 文件或删除后再次运行此脚本。")
            return False
        return True
    return True


def read_existing_env() -> dict[str, str]:
    """读取现有 .env 文件内容"""
    config: dict[str, str] = {}
    if ENV_FILE.exists():
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    config[key.strip()] = value.strip()
    return config


def select_provider() -> tuple[str, str]:
    """引导用户选择 LLM Provider"""
    print("  请选择 LLM Provider:")
    print()
    for num, info in PROVIDERS.items():
        print(f"    {num}. {info['name']}")
    print()

    while True:
        choice = input("  请输入选项 (1-3): ").strip()
        if choice in PROVIDERS:
            return PROVIDERS[choice]["key"], PROVIDERS[choice]["env_key"]
        print("  ⚠️  无效选项，请输入 1-3")


def get_api_key(env_key: str) -> str:
    """获取用户输入的 API Key"""
    print()
    print(f"  请输入你的 {env_key}（可在对应控制台找到）:")
    print(f"    OpenAI: https://platform.openai.com/api-keys")
    print(f"    Anthropic: https://console.anthropic.com/keys")
    print(f"    Groq: https://console.groq.com/keys")
    print()

    while True:
        key = input("  API Key: ").strip()
        if key:
            return key
        print("  ⚠️  API Key 不能为空")


def get_model_choice(provider: str) -> str:
    """根据 provider 推荐模型"""
    models = {
        "openai": ("gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"),
        "anthropic": ("claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-haiku-4-20250514"),
        "groq": ("qwen-2.5-7b", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"),
    }
    default = models.get(provider, models["openai"])[0]
    print(f"\n  推荐模型: {default}")
    model = input("  直接回车使用推荐，或输入自定义模型名: ").strip()
    return model if model else default


def build_env_content(
    provider: str,
    env_key: str,
    api_key: str,
    model: str,
) -> str:
    """构建 .env 文件内容"""
    return f"""# AI English Tutor — 自动生成于 {Path.cwd()}
# 由 setup.py 配置向导生成，请勿手动编辑

# === LLM Provider 选择 ===
LLM_PROVIDER={provider}

# === {provider.upper()} ===
{env_key}={api_key}
{provider.upper()}_MODEL={model}

# === 应用配置 ===
APP_HOST=0.0.0.0
APP_PORT=8000

# === LLM 开关 ===
# 设为 True 启用真实 LLM 调用；False 使用内置 mock 回复
LLM_ENABLED=False
LLM_MODE_CONVERSATION=False
LLM_MODE_CORRECTION=False
LLM_MODE_SCORING=False
"""


def write_env(content: str):
    """写入 .env 文件"""
    with open(ENV_FILE, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n  ✅ 配置文件已创建: {ENV_FILE}")


def print_next_steps():
    print()
    print("-" * 60)
    print("  下一步:")
    print()
    print("  1. 编辑 .env 文件，将 LLM_ENABLED=True 启用真实 LLM")
    print("  2. 安装依赖:")
    print("     pip install -r requirements.txt")
    print("  3. 启动后端 API:")
    print("     uvicorn api.main:app --reload --port 8000")
    print("  4. 启动 Web UI (新终端):")
    print("     streamlit run ui/main.py --server.port 8501")
    print("  5. 运行测试:")
    print("     pytest tests/ -v")
    print()
    print("-" * 60)
    print()


def main():
    print_banner()

    if not check_existing_env():
        sys.exit(0)

    existing = read_existing_env()

    # Provider 选择
    provider, env_key = select_provider()

    # API Key
    api_key = get_api_key(env_key)

    # Model
    model = get_model_choice(provider)

    # Build and write
    content = build_env_content(provider, env_key, api_key, model)
    write_env(content)

    print_next_steps()


if __name__ == "__main__":
    main()
