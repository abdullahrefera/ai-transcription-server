from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from supadata import Supadata, SupadataError
import hashlib
import time
import logging
import asyncio
from datetime import datetime
from config import settings

# Import our AI service and models
from models import AITailoringRequest, AITailoringResponse, ErrorResponse, ProductScraperRequest, ProductScraperResponse
from ai_service import AIScriptTailoringService
from scraper_service import ProductScraperService

app = FastAPI(title="Transcription & AI Script Tailoring API")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["myrefera.com", "*.myrefera.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
supadata = Supadata(api_key=settings.SUPADATA_API_KEY)
ai_service = AIScriptTailoringService()
scraper_service = ProductScraperService()

# Request body model for transcription
class UrlList(BaseModel):
    urls: List[str]
    lang: str = "en"
    text: bool = True
    mode: str = "auto"

# In-memory cache: {url_hash: {data}}
CACHE = {}

# Cache expiry in seconds (optional, e.g. 1 day = 86400s)
CACHE_TTL = 86400

def _hash_url(url: str) -> str:
    """Hash URL to use as cache key."""
    return hashlib.sha256(url.encode()).hexdigest()

def _detect_platform(url: str) -> dict:
    """
    Detect the platform from URL and return platform info.
    
    Returns:
        dict: Platform information including name, support status, and recommendations
    """
    url_lower = url.lower()
    
    if 'youtube.com' in url_lower or 'youtu.be' in url_lower:
        return {
            "platform": "YouTube",
            "supported": True,
            "reliability": "high",
            "recommendation": "Fully supported and reliable"
        }
    elif 'tiktok.com' in url_lower or 'm.tiktok.com' in url_lower:
        return {
            "platform": "TikTok",
            "supported": True,
            "reliability": "low",
            "recommendation": "May fail due to rate limiting. Consider retry logic or alternative methods."
        }
    elif 'x.com' in url_lower or 'twitter.com' in url_lower:
        return {
            "platform": "X (Twitter)",
            "supported": True,
            "reliability": "low",
            "recommendation": "May fail due to access restrictions. Consider alternative methods."
        }
    elif 'instagram.com' in url_lower:
        return {
            "platform": "Instagram",
            "supported": True,
            "reliability": "medium",
            "recommendation": "Supported but may have limitations"
        }
    elif 'vimeo.com' in url_lower:
        return {
            "platform": "Vimeo",
            "supported": False,
            "reliability": "none",
            "recommendation": "Not supported by Supadata. Use alternative transcription service."
        }
    elif 'twitch.tv' in url_lower:
        return {
            "platform": "Twitch",
            "supported": False,
            "reliability": "none",
            "recommendation": "Not supported by Supadata. Use alternative transcription service."
        }
    else:
        return {
            "platform": "Unknown",
            "supported": False,
            "reliability": "unknown",
            "recommendation": "Platform not recognized. Check if URL is valid and supported."
        }

async def _transcribe_with_retry(url: str, lang: str, text: bool, mode: str, max_retries: int = 3) -> dict:
    """
    Transcribe URL with retry logic and proper error handling.
    
    Args:
        url: Video URL to transcribe
        lang: Language code
        text: Whether to return text
        mode: Transcription mode
        max_retries: Maximum number of retry attempts
    
    Returns:
        dict: Transcription result or error information
    """
    # Detect platform and check support
    platform_info = _detect_platform(url)
    logger.info(f"🔍 Platform detected: {platform_info['platform']} (reliability: {platform_info['reliability']})")
    
    # If platform is not supported, return early with helpful message
    if not platform_info['supported']:
        logger.warning(f"⚠️ Platform {platform_info['platform']} is not supported by Supadata")
        return {
            "url": url,
            "status": "error",
            "platform": platform_info['platform'],
            "error": {
                "code": "PLATFORM_NOT_SUPPORTED",
                "message": f"Platform {platform_info['platform']} is not supported by Supadata",
                "details": platform_info['recommendation']
            }
        }
    
    # Adjust retry strategy based on platform reliability
    if platform_info['reliability'] == 'low':
        max_retries = max(max_retries, 5)  # More retries for unreliable platforms
        logger.info(f"🔄 Using enhanced retry strategy for {platform_info['platform']} (max {max_retries} attempts)")
    
    for attempt in range(max_retries):
        try:
            logger.info(f"🔄 Attempt {attempt + 1}/{max_retries} for URL: {url}")
            
            # Call Supadata API
            transcript = supadata.transcript(
                url=url,
                lang=lang,
                text=text,
                mode=mode
            )
            
            # Process successful response
            if hasattr(transcript, "content"):
                result = {
                    "url": url,
                    "platform": platform_info['platform'],
                    "language": transcript.lang,
                    "transcript": transcript.content,
                    "status": "success"
                }
                logger.info(f"✅ Transcription successful for {platform_info['platform']} URL: {url}")
                return result
            else:
                result = {
                    "url": url,
                    "platform": platform_info['platform'],
                    "job_id": transcript.job_id,
                    "status": "processing"
                }
                logger.info(f"⏳ Transcription queued for {platform_info['platform']} URL: {url}")
                return result
                
        except SupadataError as e:
            error_msg = str(e)
            logger.warning(f"⚠️ Supadata error on attempt {attempt + 1}: {error_msg}")
            
            # Check if it's a rate limiting error (429) or server error (500)
            if "status 429" in error_msg:
                # Rate limiting - wait longer before retry
                wait_time = (2 ** attempt) * 5  # Exponential backoff: 5s, 10s, 20s
                logger.info(f"⏰ Rate limited, waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)
            elif "status 400" in error_msg or "status 500" in error_msg:
                # Server error or bad request - shorter wait
                wait_time = (2 ** attempt) * 2  # Exponential backoff: 2s, 4s, 8s
                logger.info(f"⏰ Server error, waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)
            else:
                # Other errors - don't retry
                logger.error(f"❌ Non-retryable error: {error_msg}")
                return {
                    "url": url,
                    "platform": platform_info['platform'],
                    "status": "error",
                    "error": {
                        "code": "TRANSCRIPTION_ERROR",
                        "message": f"Failed to transcribe {platform_info['platform']} video: {error_msg}",
                        "details": error_msg,
                        "platform_recommendation": platform_info['recommendation']
                    }
                }
            
            # If this was the last attempt, return error
            if attempt == max_retries - 1:
                logger.error(f"❌ All retry attempts failed for {platform_info['platform']} URL: {url}")
                return {
                    "url": url,
                    "platform": platform_info['platform'],
                    "status": "error",
                    "error": {
                        "code": "TRANSCRIPTION_FAILED",
                        "message": f"Failed to transcribe {platform_info['platform']} video after {max_retries} attempts: {error_msg}",
                        "details": error_msg,
                        "platform_recommendation": platform_info['recommendation']
                    }
                }
                
        except Exception as e:
            logger.error(f"❌ Unexpected error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                return {
                    "url": url,
                    "platform": platform_info['platform'],
                    "status": "error",
                    "error": {
                        "code": "UNEXPECTED_ERROR",
                        "message": f"Unexpected error transcribing {platform_info['platform']} video: {str(e)}",
                        "details": str(e),
                        "platform_recommendation": platform_info['recommendation']
                    }
                }
            await asyncio.sleep(2 ** attempt)  # Simple exponential backoff
    
    # This should never be reached, but just in case
    return {
        "url": url,
        "platform": platform_info['platform'],
        "status": "error",
        "error": {
            "code": "UNKNOWN_ERROR",
            "message": f"Unknown error occurred during {platform_info['platform']} transcription",
            "details": "Maximum retries exceeded",
            "platform_recommendation": platform_info['recommendation']
        }
    }

# ===== EXISTING TRANSCRIPTION ENDPOINT =====
@app.post("/transcribe")
async def transcribe(data: UrlList):
    results = []
    now = time.time()
    
    logger.info(f"🎯 Starting transcription for {len(data.urls)} URLs")

    for url in data.urls:
        key = _hash_url(url)

        # ✅ Check cache
        if key in CACHE:
            entry = CACHE[key]
            # Validate TTL
            if now - entry["timestamp"] < CACHE_TTL:
                logger.info(f"📋 Using cached result for URL: {url}")
                results.append(entry["result"])
                continue
            else:
                # Expired, remove
                logger.info(f"🗑️ Cache expired for URL: {url}")
                del CACHE[key]

        # 🔄 If not cached or expired, call API with retry logic
        logger.info(f"🔄 Processing URL: {url}")
        result = await _transcribe_with_retry(
            url=url,
            lang=data.lang,
            text=data.text,
            mode=data.mode
        )

        # ✅ Save to cache (only if successful)
        if result.get("status") in ["success", "processing"]:
            CACHE[key] = {"result": result, "timestamp": now}
        
        results.append(result)

    # Count successful vs failed transcriptions
    successful = sum(1 for r in results if r.get("status") in ["success", "processing"])
    failed = len(results) - successful
    
    logger.info(f"📊 Transcription complete: {successful} successful, {failed} failed")
    
    return {
        "results": results,
        "summary": {
            "total": len(results),
            "successful": successful,
            "failed": failed
        }
    }

# ===== NEW AI SCRIPT TAILORING ENDPOINT =====
@app.post("/api/ai-tailor-script", response_model=AITailoringResponse)
async def ai_tailor_script(request: AITailoringRequest):
    """
    AI Script Tailoring endpoint that takes a transcript and product details
    and returns a marketing psychology-optimized script using GPT-5.
    """
    try:
        logger.info(f"🎯 Starting AI script tailoring for product description: {request.productDescription[:50]}...")
        logger.info(f"📝 Transcript length: {len(request.originalTranscript)} characters")
        
        # Generate tailored script using AI service
        tailoring_data = await ai_service.generate_tailored_script(request)
        
        # Create response with metadata
        response = AITailoringResponse(
            success=True,
            data=tailoring_data,
            metadata={
                "originalLength": len(request.originalTranscript.split()),
                "improvementAreas": ["product_alignment", "psychological_triggers"],
                "apiVersion": "2.0",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "model_used": ai_service.primary_model
            }
        )
        
        logger.info(f"✅ AI script tailoring completed successfully")
        logger.info(f"📊 Response Summary:")
        logger.info(f"   - Tailored Script Length: {len(tailoring_data.tailoredScript)} characters")
        logger.info(f"   - Word Count: {tailoring_data.wordCount} words")
        logger.info(f"   - Processing Time: {tailoring_data.processingTime:.2f}s")
        logger.info(f"   - Confidence: {tailoring_data.confidence}")
        logger.info(f"   - Sections: {len(tailoring_data.sectionBreakdown)}")
        logger.info(f"📝 Tailored Script Preview: {tailoring_data.tailoredScript[:100]}...")
        
        # Log section breakdown details
        if tailoring_data.sectionBreakdown:
            logger.info(f"📋 Section Breakdown:")
            for i, section in enumerate(tailoring_data.sectionBreakdown, 1):
                logger.info(f"   {i}. {section.sectionName} - {section.triggerEmotionalState}")
                logger.info(f"      Timestamp: {section.timestamp}")
                logger.info(f"      Quote: {section.originalQuote[:50]}...")
                logger.info(f"      Rewritten: {section.rewrittenVersion[:50]}...")
        
        # Log full response data for debugging
        logger.info(f"📋 Full Response Data:")
        logger.info(f"   - Success: {response.success}")
        logger.info(f"   - Model Used: {response.metadata.get('model_used', 'unknown')}")
        logger.info(f"   - API Version: {response.metadata.get('apiVersion', 'unknown')}")
        logger.info(f"   - Timestamp: {response.metadata.get('timestamp', 'unknown')}")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Error in AI script tailoring: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": {
                    "code": "AI_PROCESSING_ERROR",
                    "message": f"Failed to generate tailored script: {str(e)}"
                }
            }
        )

# ===== PLATFORM SUPPORT CHECK ENDPOINT =====
@app.get("/platform-support")
async def check_platform_support(url: str):
    """
    Check if a platform is supported and get reliability information.
    
    Args:
        url: Video URL to check platform support for
    
    Returns:
        dict: Platform support information and recommendations
    """
    platform_info = _detect_platform(url)
    return {
        "url": url,
        "platform_info": platform_info,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

# ===== HEALTH CHECK ENDPOINT =====
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Transcription & AI Script Tailoring API",
        "version": "2.1",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "openai_model": ai_service.primary_model,
        "services": ["transcription", "ai_script_tailoring", "platform_detection", "product_scraping"],
        "supported_platforms": {
            "high_reliability": ["YouTube"],
            "medium_reliability": ["Instagram"],
            "low_reliability": ["TikTok", "X (Twitter)"],
            "not_supported": ["Vimeo", "Twitch"]
        }
    }

# ===== PRODUCT SCRAPER ENDPOINT =====
@app.post("/api/scrape-product", response_model=ProductScraperResponse)
async def scrape_product(request: ProductScraperRequest):
    """
    Product scraper endpoint that extracts product description from a given URL.
    """
    try:
        logger.info(f"🔍 Starting product scraping for URL: {request.url}")
        
        # Scrape product description
        scraper_data = await scraper_service.scrape_product(request)
        
        # Create response with metadata
        response = ProductScraperResponse(
            success=True,
            data=scraper_data,
            metadata={
                "url": request.url,
                "domain": request.url.split('/')[2] if '://' in request.url else "unknown",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "apiVersion": "1.0"
            }
        )
        
        logger.info(f"✅ Product scraping completed successfully")
        logger.info(f"📊 Response Summary:")
        logger.info(f"   - Description Length: {len(scraper_data.description)} characters")
        logger.info(f"   - Title: {scraper_data.title or 'Not found'}")
        logger.info(f"📝 Description Preview: {scraper_data.description[:100]}...")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Error in product scraping: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": {
                    "code": "SCRAPING_ERROR",
                    "message": f"Failed to scrape product: {str(e)}"
                }
            }
        )
