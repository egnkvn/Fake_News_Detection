import scrapy
from urllib.parse import urljoin

class BbcInternationalSpider(scrapy.Spider):
    name = "bbc_international"
    allowed_domains = ["bbc.com"]
    start_urls = ["https://www.bbc.com/zhongwen/topics/c83plve5vmjt/trad"]

    def parse(self, response):
        # Select news article links on the topic page
        # Updated selector to include the new link format
        for article in response.css('a.gs-c-promo-heading, a.bbc-1i4ie53.e1d658bg0'):
            link = urljoin(response.url, article.attrib['href'])
            self.logger.debug(f"Found article link: {link}") # Use Scrapy logger
            yield scrapy.Request(link, callback=self.parse_article)

        # Follow pagination if available
        # Updated selector for the "Next Page" link
        # Now handles both "下一頁" (Traditional) and "后页" (Simplified) next page links
                # Identify and follow the "Next Page" link
        next_page = None
        for a in response.css('a.bbc-1spja2a'):
            span_text = a.css('span span::text').get()
            if span_text and span_text.strip() in ["后页", "下一頁"]:
                next_page = a.attrib.get('href')
                break

        if next_page:
            next_url = urljoin(response.url, next_page)
            yield scrapy.Request(next_url, callback=self.parse)

    def parse_article(self, response):
        self.logger.debug(f"Parsing article: {response.url}")
        item = {
            "title": response.css("h1::text").get(),
            "url": response.url,
            "date": response.css("time::attr(datetime)").get(),
            "author": response.css("span.bbc-1o0gmgs::text").get() if response.css("span.bbc-1o0gmgs::text").get() else response.css("span.ssrcss-1n7hynb-Contributor::text").get(),
            "content": " ".join(response.xpath('//main//p//text()').getall())
        }
        self.logger.debug(f"Scraped item: {item['title']}")
        yield item

