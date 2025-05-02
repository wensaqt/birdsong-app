from duckduckgo_search import DDGS

class WebScraper:
    def __init__(self):
        self.ddgs = DDGS()

    def get_image_url(self, bird_name: str) -> str | None:
        with self.ddgs:
            results = self.ddgs.images(keywords=f"{bird_name} bird", max_results=1)
            for result in results:
                return result["image"]
        return None