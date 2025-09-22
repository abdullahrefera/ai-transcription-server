import requests
import logging
import time
from typing import Dict, Any, Optional
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re
from models import ProductScraperRequest, ProductScraperData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductScraperService:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # Focused selectors for product description extraction
        self.description_selectors = [
            # E-commerce specific selectors
            '[data-testid="product-description"]',
            '.product-description',
            '.product-details',
            '.product-info',
            '.description',
            '[class*="description"]',
            '[class*="product-details"]',
            '[class*="product-info"]',
            
            # Meta description as fallback
            'meta[name="description"]',
            
            # Common product page selectors
            '.product-summary',
            '.product-overview',
            '.product-content',
            '.product-text',
            '.item-description',
            '.product-specs',
            
            # Generic content selectors
            'p[class*="description"]',
            'div[class*="description"]',
            'section[class*="description"]',
            
            # JSON-LD structured data
            'script[type="application/ld+json"]'
        ]
        
        self.title_selectors = [
            'h1[data-testid="product-title"]',
            'h1.product-title',
            'h1[class*="title"]',
            'h1[class*="name"]',
            'h1',
            '.product-name',
            '.product-title',
            '[data-testid="product-title"]',
            'title'
        ]

    def _is_valid_url(self, url: str) -> bool:
        """Validate if the URL is properly formatted"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False

    def _get_domain(self, url: str) -> str:
        """Extract domain from URL"""
        return urlparse(url).netloc

    def _make_request(self, url: str, timeout: int = 10) -> Optional[requests.Response]:
        """Make HTTP request with error handling"""
        try:
            logger.info(f"🌐 Fetching URL: {url}")
            response = self.session.get(url, timeout=timeout, allow_redirects=True)
            response.raise_for_status()
            logger.info(f"✅ Successfully fetched URL. Status: {response.status_code}")
            return response
        except requests.exceptions.Timeout:
            logger.error(f"⏰ Timeout fetching URL: {url}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error fetching URL {url}: {e}")
            return None

    def _extract_text_by_selectors(self, soup: BeautifulSoup, selectors: list) -> Optional[str]:
        """Extract text using multiple CSS selectors"""
        for selector in selectors:
            try:
                element = soup.select_one(selector)
                if element:
                    # Handle meta tags differently
                    if element.name == 'meta':
                        return element.get('content', '').strip()
                    else:
                        return element.get_text(strip=True)
            except Exception as e:
                logger.debug(f"Selector '{selector}' failed: {e}")
                continue
        return None

    def _extract_structured_data_description(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract description from JSON-LD structured data"""
        try:
            json_ld_scripts = soup.find_all('script', type='application/ld+json')
            for script in json_ld_scripts:
                try:
                    import json
                    data = json.loads(script.string)
                    
                    # Handle both single objects and arrays
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and 'description' in item:
                                return item['description']
                    elif isinstance(data, dict) and 'description' in data:
                        return data['description']
                        
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            logger.debug(f"Error extracting structured data: {e}")
        
        return None

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common unwanted patterns
        unwanted_patterns = [
            r'^\s*[•\-*]\s*',  # Bullet points at start
            r'\s*[•\-*]\s*$',  # Bullet points at end
            r'^\s*\d+\.\s*',   # Numbered lists
        ]
        
        for pattern in unwanted_patterns:
            text = re.sub(pattern, '', text)
        
        return text.strip()

    async def scrape_product(self, request: ProductScraperRequest) -> ProductScraperData:
        """Main method to scrape product description"""
        start_time = time.time()
        
        try:
            # Validate URL
            if not self._is_valid_url(request.url):
                raise ValueError("Invalid URL format")
            
            # Make HTTP request
            response = self._make_request(request.url)
            if not response:
                raise Exception("Failed to fetch the webpage")
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract description using multiple methods
            description = None
            
            # Method 1: Try CSS selectors
            description = self._extract_text_by_selectors(soup, self.description_selectors)
            
            # Method 2: Try structured data if CSS selectors failed
            if not description:
                description = self._extract_structured_data_description(soup)
            
            # Method 3: Fallback to meta description
            if not description:
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                if meta_desc and meta_desc.get('content'):
                    description = meta_desc.get('content')
            
            # Clean and validate description
            description = self._clean_text(description) if description else "Product description not available"
            
            # Extract title (optional, for context)
            title = self._extract_text_by_selectors(soup, self.title_selectors)
            title = self._clean_text(title) if title else None
            
            processing_time = time.time() - start_time
            
            logger.info(f"✅ Product description scraping completed in {processing_time:.2f}s")
            logger.info(f"📊 Extracted: Title={bool(title)}, Description length={len(description)} chars")
            
            return ProductScraperData(
                description=description,
                title=title
            )
            
        except Exception as e:
            logger.error(f"❌ Error scraping product description: {e}")
            processing_time = time.time() - start_time
            
            # Return error response in expected format
            return ProductScraperData(
                description=f"Error occurred while scraping the product description: {str(e)}",
                title="Error: Failed to scrape product"
            )
