import requests
from bs4 import BeautifulSoup
from newspaper import Article
from readability import Document
from urllib.parse import urlparse
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BlogScraper:
    def __init__(self, timeout: int, user_agent: str):
        self.timeout = timeout
        self.headers = {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }

    def scrape_blog_post(self, url: str) -> dict[str, str] | None:
        """
        Scrape a blog post using multiple methods for best results.
        Returns structured content or None if extraction fails.
        """
        try:
            article_data = self._extract_with_newspaper(url)
            if article_data and self._is_valid_content(article_data["content"]):
                return article_data

            readability_data = self._extract_with_readability(url)
            if readability_data and self._is_valid_content(readability_data["content"]):
                return readability_data

            basic_data = self._extract_with_beautifulsoup(url)
            if basic_data and self._is_valid_content(basic_data["content"]):
                return basic_data

            logger.warning(f"All extraction methods failed for {url}")
            return None

        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return None

    def _extract_with_newspaper(self, url: str) -> dict[str, str] | None:
        try:
            article = Article(url)
            article.download()
            article.parse()

            publish_date = None
            if article.publish_date:
                publish_date = article.publish_date.isoformat()

            return {
                "url": url,
                "title": article.title or self._extract_title_from_url(url),
                "content": article.text,
                "authors": article.authors,
                "publish_date": publish_date,
                "summary": article.summary if hasattr(article, "summary") else "",
                "tags": list(article.tags) if article.tags else [],
                "extraction_method": "newspaper3k",
                "scraped_at": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.debug(f"Newspaper extraction failed for {url}: {str(e)}")
            return None

    def _extract_with_readability(self, url: str) -> dict[str, str] | None:
        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()

            doc = Document(response.text)
            soup = BeautifulSoup(doc.content(), "html.parser")

            content = soup.get_text(separator="\n", strip=True)

            original_soup = BeautifulSoup(response.text, "html.parser")
            title = self._extract_title(original_soup) or doc.title()

            return {
                "url": url,
                "title": title or self._extract_title_from_url(url),
                "content": content,
                "authors": self._extract_authors(original_soup),
                "publish_date": self._extract_publish_date(original_soup),
                "summary": content[:500] + "..." if len(content) > 500 else content,
                "tags": self._extract_tags(original_soup),
                "extraction_method": "readability",
                "scraped_at": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.debug(f"Readability extraction failed for {url}: {str(e)}")
            return None

    def _extract_with_beautifulsoup(self, url: str) -> dict[str, str] | None:
        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                script.decompose()

            content_selectors = [
                "article",
                "main",
                ".content",
                ".post-content",
                ".entry-content",
                ".post-body",
                ".article-content",
            ]

            content = ""
            for selector in content_selectors:
                content_element = soup.select_one(selector)
                if content_element:
                    content = content_element.get_text(separator="\n", strip=True)
                    break

            if not content:
                body = soup.find("body")
                if body:
                    content = body.get_text(separator="\n", strip=True)

            return {
                "url": url,
                "title": self._extract_title(soup) or self._extract_title_from_url(url),
                "content": content,
                "authors": self._extract_authors(soup),
                "publish_date": self._extract_publish_date(soup),
                "summary": content[:500] + "..." if len(content) > 500 else content,
                "tags": self._extract_tags(soup),
                "extraction_method": "beautifulsoup",
                "scraped_at": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.debug(f"BeautifulSoup extraction failed for {url}: {str(e)}")
            return None

    def _extract_title(self, soup: BeautifulSoup) -> str | None:
        title_selectors = [
            "h1.title",
            "h1.post-title",
            "h1.entry-title",
            "h1.article-title",
            ".post-title h1",
            "h1",
        ]

        for selector in title_selectors:
            title_element = soup.select_one(selector)
            if title_element and title_element.get_text(strip=True):
                return title_element.get_text(strip=True)

        title_tag = soup.find("title")
        if title_tag:
            return title_tag.get_text(strip=True)

        return None

    def _extract_authors(self, soup: BeautifulSoup) -> list[str]:
        author_selectors = [
            ".author",
            ".byline",
            ".post-author",
            '[rel="author"]',
            ".entry-author",
        ]

        authors = []
        for selector in author_selectors:
            author_elements = soup.select(selector)
            for element in author_elements:
                author_text = element.get_text(strip=True)
                if author_text and author_text not in authors:
                    authors.append(author_text)

        return authors

    def _extract_publish_date(self, soup: BeautifulSoup) -> str | None:
        date_selectors = [
            "time[datetime]",
            ".publish-date",
            ".post-date",
            ".entry-date",
            '[property="article:published_time"]',
        ]

        for selector in date_selectors:
            date_element = soup.select_one(selector)
            if date_element:
                datetime_attr = date_element.get("datetime")
                if datetime_attr:
                    return datetime_attr

                content_attr = date_element.get("content")
                if content_attr:
                    return content_attr

                date_text = date_element.get_text(strip=True)
                if date_text:
                    return date_text

        return None

    def _extract_tags(self, soup: BeautifulSoup) -> list[str]:
        tag_selectors = [
            ".tags a",
            ".post-tags a",
            ".entry-tags a",
            ".categories a",
            '[rel="tag"]',
        ]

        tags = []
        for selector in tag_selectors:
            tag_elements = soup.select(selector)
            for element in tag_elements:
                tag_text = element.get_text(strip=True)
                if tag_text and tag_text not in tags:
                    tags.append(tag_text)

        return tags

    def _extract_title_from_url(self, url: str) -> str:
        parsed = urlparse(url)
        path = parsed.path.strip("/").replace("-", " ").replace("_", " ")
        if path:
            return path.split("/")[-1].title()
        return parsed.netloc

    def _is_valid_content(self, content: str) -> bool:
        if not content or len(content.strip()) < 100:
            return False

        # Check for common error indicators
        error_indicators = [
            "access denied",
            "page not found",
            "404 error",
            "javascript required",
            "enable javascript",
        ]

        content_lower = content.lower()
        for indicator in error_indicators:
            if indicator in content_lower:
                return False

        return True
