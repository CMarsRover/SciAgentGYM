import http.client
import json
from typing import List, Dict, Optional
from dataclasses import dataclass 


@dataclass
class SearchResult:
    """搜索结果数据类"""
    title: str
    link: str
    snippet: str
    source: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "link": self.link,
            "snippet": self.snippet,
            "source": self.source
        }
    
    def format_for_llm(self) -> str:
        """格式化为适合LLM理解的文本"""
        return f"标题: {self.title}\n链接: {self.link}\n摘要: {self.snippet}\n来源: {self.source}\n"

def search_with_serpapi(
                query: str, 
                num_results: int) -> List[SearchResult]: 

      conn = http.client.HTTPSConnection("google.serper.dev")
      payload = json.dumps({
        "q": query , #"pubchem数据库访问方式",
        "num_results":num_results
      })
      headers = {
        'X-API-KEY': '1ad52639b8a625855aae160ee81576a57e08ccb1',
        'Content-Type': 'application/json'
      }
      conn.request("POST", "/search", payload, headers)
      res = conn.getresponse()
    
      data = json.loads(res.read().decode("utf-8")) 
      results = []
      for item in data.get("organic", []):
          result = SearchResult(
              title=item.get("title", ""),
              link=item.get("link", ""),
              snippet=item.get("snippet", ""),
              source="SerpAPI"
          )
          results.append(result) 
      return results



