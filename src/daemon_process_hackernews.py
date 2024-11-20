import schedule  # 导入 schedule 实现定时任务执行器
import time  # 导入time库，用于控制时间间隔
import signal  # 导入signal库，用于信号处理
import sys  # 导入sys库，用于执行系统相关的操作
from config import Config  # 导入配置管理类
from notifier import Notifier  # 导入通知器类，用于发送通知
from report_generator import ReportGenerator  # 导入报告生成器类
from llm import LLM  # 导入语言模型类，可能用于生成报告内容
from subscription_manager import SubscriptionManager  # 导入订阅管理器类，管理GitHub仓库订阅
from logger import LOG  # 导入日志记录器

# 环境变量导入
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# 以下依赖用于爬取 Hacker News 网站内容
import requests
import os
from bs4 import BeautifulSoup
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI, ChatOpenAI
from datetime import datetime

def graceful_shutdown(signum, frame):
    # 优雅关闭程序的函数，处理信号时调用
    LOG.info("[优雅退出]守护进程接收到终止信号")
    sys.exit(0)  # 安全退出程序


class NewsItem:
    def __init__(self, title, link, comments):
        self.title = title
        self.link = link
        self.comments = comments

    def __repr__(self):
        return f"NewsItem(title={self.title!r}, link={self.link!r}, comments={self.comments!r})"


class HackerNewsScraper:
    def __init__(self, url="https://news.ycombinator.com/"):
        self.url = url

    def fetch_news(self):
        response = requests.get(self.url)

        # Check for successful response
        if response.status_code != 200:
            print("Failed to fetch the website.")
            return []

        # Parse the content with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        news_items = []

        # Extract news items and comments
        titles = soup.select('.titleline > a')
        subtexts = soup.select('.subtext')

        for title, subtext in zip(titles, subtexts):
            # Extract comments count
            comment_tag = subtext.find_all('a')[-1]
            if "comment" in comment_tag.text:
                try:
                    comments = int(comment_tag.text.split()[0])  # Extract the number before "comments"
                except ValueError:
                    comments = 0
            else:
                comments = 0

            # Append to news items list
            news_items.append(NewsItem(title.text, title['href'], comments))

        return news_items

    def sort_news_by_comments(self, news_items):
        return sorted(news_items, key=lambda item: item.comments, reverse=True)


class HackerNewsAnalyzer:
    def __init__(self, news_list, llm=None):
        self.news_list = news_list
        self.llm = ChatOpenAI(
            api_key=os.environ['OPENAI_API_KEY'],
            model_name=os.environ['OPENAI_GPT_MODEL'],
            # http_client=httpx.Client(proxies=os.environ['HTTP_PROXY']),
        )

    def analyze(self) -> str :
        # 获取当天日期
        current_date = datetime.now().strftime("%Y-%m-%d")

        # 构建新闻内容文本
        news_content = "\n".join(
            [f"{i+1}. Title: {item.title}, Link: {item.link},  comments count: {item.comments} " for i, item in enumerate(self.news_list)])

        # 构建 Prompt
        hk_prompt = f"""
你是一个关注 Hacker News 的技术专家，擅于洞察技术热点和发展趋势。

任务：
你收到的 Hacker News Top List 中,每条News里面都包含标题、链接和评论数, 评论数通常代表着热度。请分析和总结当前技术圈讨论的热点，输出洞察。

输出格式：
# Hacker News 技术洞察
## 时间：{current_date}
## 技术前沿趋势与热点话题
1. **个人项目与创作**：许多用户在 "Ask HN" 讨论中分享了他们正在进行的项目，这凸显了开发者界对个人创作及创业的持续热情。
2. **网络安全思考**：有关于“防守者和攻击者思考方式”的讨论引发了对网络安全策略的深入思考。这种对比强调防守与攻击之间的心理与技术差异，表明网络安全领域对攻击者策略的关注日益增加。

以下是 Hacker News Top List：
{news_content}
"""
        # 使用 LLM 生成总结
        prompt = PromptTemplate.from_template(hk_prompt)
        LOG.info(prompt.format())

        # 调用 OpenAI GPT 模型生成报告
        result = self.llm.invoke(prompt.format()).content

        return result


def hackernews_job(notifier):
    LOG.info("[执行定时任务--开始]")
    scraper = HackerNewsScraper()
    news_list = scraper.fetch_news()

    if news_list:
        # Sort news by comments
        sorted_news = scraper.sort_news_by_comments(news_list)

        # 分析新闻
        analyzer = HackerNewsAnalyzer(sorted_news[:5])
        report_content = analyzer.analyze()
        LOG.info(report_content)

        # 保存报告
        report_file_path = f"daily_progress/hackernews_{datetime.now().strftime('%Y-%m-%d')}_report.md"
        with open(report_file_path, 'w+') as report_file:
            report_file.write(report_content)  # 写入生成的报告
        LOG.info(f"GitHub 项目报告已保存到 {report_file_path}")

        # 邮件发送
        # notifier.notify("Hacker News", report_content)
    else:
        LOG.info("本次没有获取到新闻信息")

    LOG.info("[执行定时任务--结束]")

def main():
    config = Config()  # 创建配置实例
    notifier = Notifier(config.email)  # 创建通知器实例

    # 设置信号处理器
    signal.signal(signal.SIGTERM, graceful_shutdown)
    hackernews_job(notifier)

    # 安排每天的定时任务
    schedule.every(config.freq_days).days.at(
        config.exec_time
    ).do(hackernews_job, notifier, config.freq_days)

    try:
        # 在守护进程中持续运行
        while True:
            schedule.run_pending()
            time.sleep(10)  # 短暂休眠以减少 CPU 使用
    except Exception as e:
        LOG.error(f"主进程发生异常: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
