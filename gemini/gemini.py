import re
from decouple import config
import google.generativeai as genai

GEMINI_API_KEY = config('GEMINI_API_KEY')


def getgemini(resume):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        f"Rate my resume out of 10 in the format RATE/10, and add improvements in bullet points: {resume}")
    generated_text = response.candidates[0].content.parts[0].text
    return generated_text


def parse_gemini_response(response):
    try:
        rating_match = re.search(r'RATE:\s*(\d+)/10', response)
        rating = rating_match.group(1) if rating_match else "N/A"

        improvements_match = re.search(r'\*\*Improvements:\*\*\n\n([\s\S]+)', response)
        improvements_text = improvements_match.group(1) if improvements_match else "No improvements provided."
        improvements = improvements_text.strip().split("\n\n* ")

        return rating, improvements
    except Exception as e:
        return "N/A", [f"Error parsing response: {str(e)}"]

