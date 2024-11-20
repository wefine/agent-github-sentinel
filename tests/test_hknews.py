# import requests
# from bs4 import BeautifulSoup
#
# class NewsItem:
#     def __init__(self, title, link, comments):
#         self.title = title
#         self.link = link
#         self.comments = comments
#
#     def __repr__(self):
#         return f"NewsItem(title={self.title!r}, link={self.link!r}, comments={self.comments!r})"
#
# class HackerNewsScraper:
#     def __init__(self, url="https://news.ycombinator.com/"):
#         self.url = url
#
#     def fetch_news(self):
#         response = requests.get(self.url)
#
#         # Check for successful response
#         if response.status_code != 200:
#             print("Failed to fetch the website.")
#             return []
#
#         # Parse the content with BeautifulSoup
#         soup = BeautifulSoup(response.text, 'html.parser')
#         news_items = []
#
#         # Extract news items and comments
#         titles = soup.select('.titleline > a')
#         subtexts = soup.select('.subtext')
#
#         for title, subtext in zip(titles, subtexts):
#             # Extract comments count
#             comment_tag = subtext.find_all('a')[-1]
#             if "comment" in comment_tag.text:
#                 try:
#                     comments = int(comment_tag.text.split()[0])  # Extract the number before "comments"
#                 except ValueError:
#                     comments = 0
#             else:
#                 comments = 0
#
#             # Append to news items list
#             news_items.append(NewsItem(title.text, title['href'], comments))
#
#         return news_items
#
#     def sort_news_by_comments(self, news_items):
#         return sorted(news_items, key=lambda item: item.comments, reverse=True)
#
# if __name__ == "__main__":
#     scraper = HackerNewsScraper()
#     news_list = scraper.fetch_news()
#
#     if news_list:
#         # Sort news by comments
#         sorted_news = scraper.sort_news_by_comments(news_list)
#
#         # Print sorted news
#         for i, news_item in enumerate(sorted_news, start=1):
#             print(f"{i}. {news_item.title} ({news_item.link}) - Comments: {news_item.comments}")
#     else:
#         print("No news found.")
