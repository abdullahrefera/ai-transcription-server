import json
import re
import time
import logging
import asyncio
from typing import Dict, Any
from openai import OpenAI
from config import settings
from models import AITailoringRequest, AITailoringData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIScriptTailoringService:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.primary_model = "gpt-4.1-mini"  # Only use gpt-4.1-mini
    
    def get_system_prompt(self) -> str:
        """Marketing psychology system prompt based on the migration guide"""
        return """You are a world-class marketing psychologist whose expertise blends:
• Daniel Kahneman → loss aversion, cognitive biases
• Robert Cialdini → persuasion triggers (authority, scarcity, reciprocity, etc.)
• Claude Hopkins → scientific advertising & urgency
• Rory Sutherland → reframing value & identity psychology
• Alex Hormozi → value stacking & grand slam offers
⸻
Task
Break down the following transcript into distinct psychological trigger sections with timestamps, then rewrite each section for [PRODUCT NAME + short description here].
Maintain the same timing, pacing, and structure as the original transcript.
⸻
For Each Section Provide
1. Section Name → simple label (Hook, Demonstration, Payoff, etc.)
2. Trigger / Emotional State →
• For the first section: the psychological trigger at play.
• For later sections: what the viewer feels or thinks in that moment.
3. Original Quote → exact line from transcript.
4. Rewritten Version → adapted for [PRODUCT], keeping same timing + psychological impact.
5. Scene Description → practical filming instructions for creators (iPhone + CapCut editing, natural/simple setups).
⸻
Psychological Principles to Weave In
• Loss Aversion (fear of losing / wasting) – Kahneman
• Social Proof, Authority, Scarcity, Reciprocity, Consistency – Cialdini
• Urgency & Action Clarity – Hopkins
• Identity Reinforcement, Reframing Value – Sutherland
• Value Stacking, Grand Slam Offers – Hormozi
⸻
Output Goal
• Rewrite must feel like a natural, organic recommendation, not an ad.
• Viewer should unconsciously think: "I need this" before realizing they're being marketed to.
• Each rewritten version must match the original timestamps for smooth editing.
⸻
Final Deliverables
• Full section-by-section breakdown with rewritten product script.
• A Sutherland Alchemy explanation: how reframing transforms perceived value.
• A Hormozi Value Stack breakdown: why the offer feels like a "steal."
⸻

Transcript:"If you have an iPhone,
00:00:01 --> 00:00:03 you have to do this from your phone.
00:00:03 --> 00:00:04 Go to settings,
00:00:04 --> 00:00:05 tap on Screen Time,
00:00:05 --> 00:00:07 go to content and Privacy Restrictions,
00:00:07 --> 00:00:08 click on it,
00:00:08 --> 00:00:10 scroll down and click on Passcode Changes
00:00:10 --> 00:00:12 and tap on don't allow.
00:00:12 --> 00:00:13 Do the same on account Changes,
00:00:13 --> 00:00:14 then go to lock
00:00:14 --> 00:00:17 Screen Time settings and type in different passcode
00:00:17 --> 00:00:18 from the one you use to unlock the phone.
00:00:18 --> 00:00:19 After doing this,
00:00:19 --> 00:00:21 if somebody steals your phone,
00:00:21 --> 00:00:23 they won't be able to remove your icloud account
00:00:23 --> 00:00:25 or edit the account settings.
00:00:25 --> 00:00:27 It will require for the passcode to do that."

CRITICAL: You MUST return ONLY a valid JSON object - no markdown, no explanations, no additional text. Return exactly this structure:

{
  "tailoredScript": "string - the complete tailored script",
  "confidence": 0.95,
  "improvementAreas": ["array", "of", "strings"],
  "sectionBreakdown": [
    {
      "sectionName": "Hook",
      "triggerEmotionalState": "Curiosity + Authority",
      "originalQuote": "exact quote from transcript",
      "rewrittenVersion": "rewritten version for product",
      "sceneDescription": "filming instructions",
      "psychologicalPrinciples": ["Loss Aversion", "Authority"],
      "timestamp": "00:00:01 --> 00:00:03"
    }
  ],
  "sutherlandAlchemy": {
    "explanation": "how reframing transforms value perception",
    "valueReframing": [
      {
        "original": "original perception",
        "reframed": "new perception",
        "psychologyBehind": "explanation"
      }
    ],
    "identityShifts": ["identity transformation triggers"]
  },
  "hormoziValueStack": {
    "coreOffer": "main product value proposition",
    "valueElements": [
      {
        "element": "specific benefit",
        "perceivedValue": "$X value",
        "actualCost": "$Y cost"
      }
    ],
    "totalStack": {
      "totalPerceivedValue": "$XXX",
      "actualPrice": "$YY",
      "valueMultiplier": "Xx"
    },
    "grandSlamElements": ["what makes this irresistible"]
  }
}

DO NOT include any other text outside this JSON structure."""

    def create_user_prompt(self, request: AITailoringRequest) -> str:
        """Create user prompt with transcript and product details"""
        return f"""
Transcript:
{request.originalTranscript}

Product: {request.productDescription}

Please analyze this transcript and rewrite it for the product above, following all the psychological principles and structure requirements outlined in the system prompt.
"""

    def parse_ai_response(self, ai_response: str) -> Dict[str, Any]:
        """Parse AI response with robust error handling"""
        logger.info(f"🔍 Parsing AI response. Length: {len(ai_response)}")
        
        # Debug: Log response structure analysis
        self._log_response_analysis(ai_response)
        
        # Clean response (remove markdown code blocks)
        clean_response = ai_response.strip()
        if clean_response.startswith('```json'):
            clean_response = re.sub(r'^```json\s*', '', clean_response)
            clean_response = re.sub(r'\s*```$', '', clean_response)
        elif clean_response.startswith('```'):
            clean_response = re.sub(r'^```\s*', '', clean_response)
            clean_response = re.sub(r'\s*```$', '', clean_response)

        try:
            parsed_data = json.loads(clean_response)
            logger.info("✅ JSON parsing successful")
            return parsed_data
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parsing failed: {e}")
            logger.info(f"📄 Response content (first 500 chars): {clean_response[:500]}...")
            logger.info(f"📄 Response content (last 100 chars): ...{clean_response[-100:]}")
            
            # Attempt partial data recovery
            if '"tailoredScript"' in clean_response:
                logger.info("🔄 Attempting partial data recovery...")
                try:
                    # Try to fix common JSON issues
                    fixed_response = self._fix_common_json_issues(clean_response)
                    parsed_result = json.loads(fixed_response)
                    logger.info("✅ Successfully recovered partial JSON data")
                    return parsed_result
                except json.JSONDecodeError as fix_error:
                    logger.error(f"❌ Failed to fix JSON issues: {fix_error}")
                    logger.info(f"📄 Fixed response (last 100 chars): ...{fixed_response[-100:] if len(fixed_response) > 100 else fixed_response}")
                except Exception as fix_error:
                    logger.error(f"❌ Unexpected error during JSON fix: {fix_error}")
            
            # Return minimal fallback structure
            return {
                "tailoredScript": "Error: Could not parse AI response. Please try again.",
                "confidence": 0.1,
                "improvementAreas": ["parsing_error"],
                "sectionBreakdown": [],
                "sutherlandAlchemy": {
                    "explanation": "Error in parsing response",
                    "valueReframing": [],
                    "identityShifts": []
                },
                "hormoziValueStack": {
                    "coreOffer": "Error in analysis",
                    "valueElements": [],
                    "totalStack": {},
                    "grandSlamElements": []
                }
            }

    def _fix_common_json_issues(self, response: str) -> str:
        """Attempt to fix common JSON formatting issues"""
        logger.info("🔧 Attempting to fix JSON issues...")
        
        # Fix unterminated strings first
        response = self._fix_unterminated_strings(response)
        
        # Add missing closing braces if needed
        open_braces = response.count('{')
        close_braces = response.count('}')
        if open_braces > close_braces:
            logger.info(f"🔧 Adding {open_braces - close_braces} missing closing braces")
            response += '}' * (open_braces - close_braces)
        
        # Add missing closing brackets
        open_brackets = response.count('[')
        close_brackets = response.count(']')
        if open_brackets > close_brackets:
            logger.info(f"🔧 Adding {open_brackets - close_brackets} missing closing brackets")
            response += ']' * (open_brackets - close_brackets)
        
        return response
    
    def _fix_unterminated_strings(self, response: str) -> str:
        """Fix unterminated strings in JSON response"""
        try:
            # Find the last opening quote that might be unterminated
            lines = response.split('\n')
            
            # Check if we're in the middle of a string value
            in_string = False
            escape_next = False
            last_valid_pos = 0
            
            for i, char in enumerate(response):
                if escape_next:
                    escape_next = False
                    continue
                    
                if char == '\\':
                    escape_next = True
                    continue
                    
                if char == '"':
                    in_string = not in_string
                    if not in_string:
                        last_valid_pos = i
                elif not in_string and char in '{}[],:':
                    last_valid_pos = i
            
            # If we're still in a string at the end, close it
            if in_string:
                logger.info("🔧 Detected unterminated string, attempting to close it")
                # Find the best place to terminate the string
                # Look for a logical end point (sentence ending, etc.)
                response_up_to_error = response[:last_valid_pos + 1]
                
                # Try to find a good place to end the string
                last_sentence_end = max(
                    response_up_to_error.rfind('.'),
                    response_up_to_error.rfind('!'),
                    response_up_to_error.rfind('?')
                )
                
                if last_sentence_end > 0:
                    # Truncate at sentence end and close the string
                    response = response_up_to_error[:last_sentence_end + 1] + '"'
                    logger.info(f"🔧 Truncated string at sentence end (pos {last_sentence_end})")
                else:
                    # Just close the string at the last valid position
                    response = response_up_to_error + '"'
                    logger.info(f"🔧 Closed string at last valid position (pos {last_valid_pos})")
                
                # Add any missing structural elements after the string
                if not response.rstrip().endswith('}'):
                    # We might need to add more structure
                    response = self._complete_json_structure(response)
            
            return response
            
        except Exception as e:
            logger.warning(f"⚠️ Error fixing unterminated strings: {e}")
            return response
    
    def _complete_json_structure(self, response: str) -> str:
        """Complete the JSON structure after fixing unterminated strings"""
        try:
            # Parse what we have so far to understand the structure
            json.loads(response)
            return response  # If it parses, we're good
        except json.JSONDecodeError as e:
            # Add minimal required structure to make it valid
            if '"tailoredScript"' in response and not '"confidence"' in response:
                # Add missing required fields
                response = response.rstrip().rstrip(',') + ','
                response += '''
  "confidence": 0.8,
  "improvementAreas": ["truncated_response"],
  "sectionBreakdown": [],
  "sutherlandAlchemy": {
    "explanation": "Response was truncated and recovered",
    "valueReframing": [],
    "identityShifts": []
  },
  "hormoziValueStack": {
    "coreOffer": "Analysis incomplete due to truncation",
    "valueElements": [],
    "totalStack": {},
    "grandSlamElements": []
  }
}'''
            
            return response
    
    def _log_response_analysis(self, response: str) -> None:
        """Log detailed analysis of the AI response for debugging"""
        try:
            # Count key structural elements
            open_braces = response.count('{')
            close_braces = response.count('}')
            open_brackets = response.count('[')
            close_brackets = response.count(']')
            quote_count = response.count('"')
            
            # Check if response appears to be truncated
            ends_with_valid_json = response.rstrip().endswith(('}', ']', '"'))
            
            logger.info(f"📊 Response Analysis:")
            logger.info(f"   - Length: {len(response)} characters")
            logger.info(f"   - Open braces: {open_braces}, Close braces: {close_braces}")
            logger.info(f"   - Open brackets: {open_brackets}, Close brackets: {close_brackets}")
            logger.info(f"   - Quote count: {quote_count} ({'even' if quote_count % 2 == 0 else 'odd - potential unterminated string'})")
            logger.info(f"   - Ends with valid JSON: {ends_with_valid_json}")
            logger.info(f"   - Last 50 chars: '{response[-50:]}'")
            
            # Check for required JSON fields
            required_fields = ['tailoredScript', 'confidence', 'sectionBreakdown']
            for field in required_fields:
                if f'"{field}"' in response:
                    logger.info(f"   - ✅ Found required field: {field}")
                else:
                    logger.info(f"   - ❌ Missing required field: {field}")
                    
        except Exception as e:
            logger.warning(f"⚠️ Error in response analysis: {e}")

    
    def _get_standard_params(self, model: str, system_prompt: str, user_prompt: str) -> dict:
        """Get standard chat completions API parameters - optimized for speed"""
        base_params = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": settings.TEMPERATURE,
            "max_tokens": settings.MAX_TOKENS
        }
        
        if "gpt-4o" in model or "gpt-4" in model:
            # GPT-4 and GPT-4o models
            base_params["response_format"] = {"type": "json_object"}
        else:
            # Older models like GPT-3.5
            # Don't add response_format for older models
            pass
        
        return base_params

    async def call_openai_api(self, system_prompt: str, user_prompt: str) -> str:
        """Call OpenAI API using only gpt-4.1-mini"""
        
        try:
            logger.info(f"🤖 Calling {self.primary_model} API")
            
            params = self._get_standard_params(self.primary_model, system_prompt, user_prompt)
            response = await asyncio.wait_for(
                asyncio.to_thread(self.client.chat.completions.create, **params),
                timeout=settings.REQUEST_TIMEOUT
            )
            
            content = response.choices[0].message.content
            logger.info(f"✅ {self.primary_model} API Success. Response length: {len(content)}")
            return content

        except asyncio.TimeoutError:
            logger.error(f"⏰ Timeout with {self.primary_model} after {settings.REQUEST_TIMEOUT}s")
            raise Exception(f"API call timed out after {settings.REQUEST_TIMEOUT} seconds")
        except Exception as e:
            logger.error(f"❌ Error with {self.primary_model}: {e}")
            raise e

    def calculate_metadata(self, tailored_script: str, original_transcript: str, processing_time: float) -> Dict[str, Any]:
        """Calculate metadata for the response"""
        word_count = len(tailored_script.split())
        original_length = len(original_transcript.split())
        
        # Estimate read time (average 150 words per minute)
        read_time_minutes = word_count / 150
        if read_time_minutes < 1:
            estimated_read_time = f"{int(read_time_minutes * 60)}s"
        else:
            estimated_read_time = f"{read_time_minutes:.1f}m"
        
        return {
            "wordCount": word_count,
            "estimatedReadTime": estimated_read_time,
            "originalLength": original_length,
            "processingTime": processing_time
        }

    async def generate_tailored_script(self, request: AITailoringRequest) -> AITailoringData:
        """Main method to generate tailored script"""
        start_time = time.time()
        
        try:
            # Prepare prompts
            system_prompt = self.get_system_prompt()
            user_prompt = self.create_user_prompt(request)
            
            # Call OpenAI API
            ai_response = await self.call_openai_api(system_prompt, user_prompt)
            
            # Parse response
            parsed_data = self.parse_ai_response(ai_response)
            
            # Calculate metadata
            processing_time = time.time() - start_time
            metadata = self.calculate_metadata(
                parsed_data.get("tailoredScript", ""),
                request.originalTranscript,
                processing_time
            )
            
            # Create response data
            return AITailoringData(
                tailoredScript=parsed_data.get("tailoredScript", ""),
                confidence=parsed_data.get("confidence", 0.8),
                processingTime=processing_time,
                wordCount=metadata["wordCount"],
                estimatedReadTime=metadata["estimatedReadTime"],
                sectionBreakdown=parsed_data.get("sectionBreakdown", []),
                sutherlandAlchemy=parsed_data.get("sutherlandAlchemy", {}),
                hormoziValueStack=parsed_data.get("hormoziValueStack", {})
            )
            
        except Exception as e:
            logger.error(f"❌ Error in generate_tailored_script: {e}")
            processing_time = time.time() - start_time
            
            # Return error response in expected format
            return AITailoringData(
                tailoredScript=f"Error generating script: {str(e)}",
                confidence=0.0,
                processingTime=processing_time,
                wordCount=0,
                estimatedReadTime="0s",
                sectionBreakdown=[],
                sutherlandAlchemy={
                    "explanation": "Error occurred during processing",
                    "valueReframing": [],
                    "identityShifts": []
                },
                hormoziValueStack={
                    "coreOffer": "Error in analysis",
                    "valueElements": [],
                    "totalStack": {},
                    "grandSlamElements": []
                }
            )
