import os
from typing import Dict, List, Optional
from openai import OpenAI
from prompts.xiaohongshu_template import get_complete_prompt


class XiaohongshuWriter:
    """小红书文案生成器（集成完整提示词模板）"""

    def __init__(self, api_key: str = None, base_url: str = None, model: str = "qwen-turbo"):
        """
        初始化大模型客户端

        Args:
            api_key: API密钥（优先使用传入的密钥，否则尝试从环境变量获取）
            base_url: API基础URL
            model: 模型名称
        """
        # 优先使用传入的API密钥
        if api_key:
            self.api_key = api_key
            self.api_source = "user_input"
        else:
            # 其次尝试从环境变量获取
            self.api_key = os.getenv("DASHSCOPE_API_KEY")
            self.api_source = "env_variable"

        if not self.api_key:
            raise ValueError("API密钥未设置！请提供API密钥或设置环境变量")

        # 根据模型确定base_url
        if model == "deepseek-chat":
            self.base_url = base_url or "https://api.deepseek.com"
        else:
            self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"

        self.model = model

        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        print(f"✅ 已初始化API客户端，使用模型：{self.model}，密钥来源：{self.api_source}")

    def test_connection(self):
        """测试API连接"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "请回复'连接成功'"}],
                max_tokens=10,
                temperature=0.1
            )
            return True, response.choices[0].message.content
        except Exception as e:
            return False, str(e)

    def generate_with_prompt(self, prompt: str, temperature: float = 0.7) -> str:
        """
        使用自定义提示词生成内容

        Args:
            prompt: 完整的提示词
            temperature: 创意度（0-1）
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是专业的小红书爆款文案写作专家"},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=2000,
                stream=False
            )
            return response.choices[0].message.content

        except Exception as e:
            error_msg = f"API调用失败：{str(e)}"
            print(error_msg)
            return f"❌ 生成失败：{error_msg}"

    def generate_xiaohongshu(
            self,
            subject: str,
            style: str = "活泼",
            opening_method: str = "提出疑问",
            selected_keywords: List[str] = None,
            num_titles: int = 5,
            temperature: float = 0.7
    ) -> Dict:
        """
        生成小红书文案（完整版）

        Returns:
            dict: 包含生成结果和元数据
        """
        # 构建完整提示词
        prompt = get_complete_prompt(
            subject=subject,
            style=style,
            opening_method=opening_method,
            selected_keywords=selected_keywords,
            num_titles=num_titles
        )

        # 调用API生成
        content = self.generate_with_prompt(prompt, temperature)

        # 返回结果
        return {
            "subject": subject,
            "style": style,
            "opening_method": opening_method,
            "keywords": selected_keywords or ["绝绝子", "建议收藏"],
            "content": content,
            "prompt_length": len(prompt),
            "response_length": len(content) if content else 0
        }

    def quick_generate(self, subject: str) -> str:
        """快速生成（简化版）"""
        from prompts.xiaohongshu_template import get_simple_prompt
        prompt = get_simple_prompt(subject)
        return self.generate_with_prompt(prompt)


def create_writer(provider: str = "aliyun", user_api_key: str = None) -> XiaohongshuWriter:
    """
    快速创建文案生成器

    Args:
        provider: API提供商，可选 "aliyun" 或 "deepseek"
        user_api_key: 用户提供的API密钥（优先使用）
    """
    if provider == "aliyun":
        return XiaohongshuWriter(
            api_key=user_api_key,  # 优先使用用户输入的密钥
            model="qwen-turbo",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    elif provider == "deepseek":
        return XiaohongshuWriter(
            api_key=user_api_key,  # 优先使用用户输入的密钥
            model="deepseek-chat",
            base_url="https://api.deepseek.com"
        )
    else:
        raise ValueError(f"不支持的提供商：{provider}")


# 测试代码
if __name__ == "__main__":
    # 测试API连接
    print("正在测试API连接...")

    try:
        # 使用环境变量或传入密钥
        api_key = os.getenv("DASHSCOPE_API_KEY")

        if api_key:
            writer = XiaohongshuWriter(api_key=api_key)
        else:
            print("⚠️ 未设置环境变量DASHSCOPE_API_KEY，尝试创建测试writer...")
            writer = create_writer("aliyun", None)  # 这会触发错误，因为没密钥

        success, message = writer.test_connection()

        if success:
            print(f"✅ 连接成功：{message}")

            # 测试生成文案
            print("\n正在生成小红书文案...")
            result = writer.generate_xiaohongshu(
                subject="周末咖啡厅自习指南",
                style="活泼",
                selected_keywords=["绝绝子", "建议收藏", "打工人"],
                temperature=0.8
            )

            print(f"\n🎯 主题：{result['subject']}")
            print(f"🎨 风格：{result['style']}")
            print(f"🔑 关键词：{', '.join(result['keywords'])}")
            print(f"📏 提示词长度：{result['prompt_length']} 字符")
            print(f"📝 生成结果：\n{result['content'][:200]}...")

        else:
            print(f"❌ 连接失败：{message}")

    except ValueError as e:
        print(f"❌ 初始化失败：{e}")
        print("请确保：")
        print("1. 已设置环境变量 DASHSCOPE_API_KEY 或传入API密钥")
        print("2. API密钥有效且有额度")
        print("3. 网络连接正常")