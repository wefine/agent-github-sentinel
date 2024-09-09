import os
from openai import OpenAI  # 导入OpenAI库用于访问GPT模型
from logger import LOG  # 导入日志模块

system_prompt = """
 你是一位资历丰富的项目经理，你会根据收到的项目的最新信息，整理输出一份中文的项目进展报告，报告以 项目名称+“项目进展” 开头，需要包含：时间范围、新增功能、主要改进，修复问题等章节内容。

一份参考示例如下:
# LangChain 项目进展
## 时间范围：2024-08-13 ~ 2024-08-18

## 新增功能
- 增加了对中文的支持
- 添加对openai函数调用的支持

## 主要改进
- 优化了流方式的处理
- 优化路由转接方式

## 修复问题
- 修复了llm代理模块的bug
- 修复了数据处理模块的bug

 """


class LLM:
    def __init__(self):
        import httpx
        # 创建一个OpenAI客户端实例
        self.client = OpenAI(
            api_key=os.environ['OPENAI_API_KEY'],
            http_client=httpx.Client(proxies=os.environ['HTTP_PROXY'])
        )
        # 配置日志文件，当文件大小达到1MB时自动轮转，日志级别为DEBUG
        LOG.add("daily_progress/llm_logs.log", rotation="1 MB", level="DEBUG")

    def generate_daily_report(self, markdown_content, dry_run=False):
        # 构建一个用于生成报告的提示文本，要求生成的报告包含新增功能、主要改进和问题修复
        prompt = f"以下是项目的最新进展，根据功能合并同类项，形成一份简报，至少包含：1）新增功能；2）主要改进；3）修复问题；:\n\n{markdown_content}"

        if dry_run:
            # 如果启用了dry_run模式，将不会调用模型，而是将提示信息保存到文件中
            LOG.info("Dry run mode enabled. Saving prompt to file.")
            with open("daily_progress/prompt.txt", "w+") as f:
                f.write(prompt)
            LOG.debug("Prompt saved to daily_progress/prompt.txt")
            return "DRY RUN"

        # 日志记录开始生成报告
        LOG.info("Starting report generation using GPT model.")

        try:
            # 调用OpenAI GPT模型生成报告
            response = self.client.chat.completions.create(
                model=os.environ["OPENAI_GPT_MODEL"],  # 指定使用的模型版本
                messages=[
                    {"role": "system", "content": system_prompt},  # 指定系统扮演角色
                    {"role": "user", "content": prompt}  # 本次用户提交的消息
                ]
            )
            LOG.debug("GPT response: {}", response)
            # 返回模型生成的内容
            return response.choices[0].message.content
        except Exception as e:
            # 如果在请求过程中出现异常，记录错误并抛出
            LOG.error("An error occurred while generating the report: {}", e)
            raise
