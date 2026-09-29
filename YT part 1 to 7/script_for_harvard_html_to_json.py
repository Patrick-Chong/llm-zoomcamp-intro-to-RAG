import json
import os
from bs4 import BeautifulSoup
from openai import OpenAI
from pydantic import BaseModel
import requests

# Initialize OpenAI client (requires OPENAI_API_KEY environment variable set)
client = OpenAI()


# 1. Define the target JSON structure using Pydantic
class FAQItem(BaseModel):
    id: str
    category: str
    question: str
    answer: str


class FAQList(BaseModel):
    items: list[FAQItem]


def generate_cs50_json():
    # 2. Scrape the live CS50 syllabus web page
    url = "https://cs50.harvard.edu/x/2026/syllabus/"
    print(f"Fetching syllabus from {url}...")

    response = requests.get(url)
    response.raise_for_status()

    # 3. Clean the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove script tags, styles, navigation bars, and footers
    for clutter in soup(["script", "style", "nav", "footer", "header", "aside"]):
        clutter.extract()

    # Extract clean text from the main body
    clean_text = soup.get_text(separator="\n", strip=True)

    print("Extracted clean body text. Passing to OpenAI for JSON parsing...")

    # 4. Use OpenAI Structured Outputs to parse into Pydantic schema
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert data structuring assistant. Extract core course policies, "
                    "grading thresholds, academic honesty rules (including the Regret Clause and AI guidelines), "
                    "resubmission rules, and final project requirements from the provided text into a clean FAQ list."
                ),
            },
            {
                "role": "user",
                "content": clean_text[:15000],  # Send the main text block
            },
        ],
        response_format=FAQList,
    )

    # 5. Extract structured data and save to cs50_faq.json
    parsed_data = completion.choices[0].message.parsed.model_dump()
    output_filename = "cs50_faq.json"

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(parsed_data["items"], f, indent=2, ensure_ascii=False)

    full_path = os.path.abspath(output_filename)
    print(
        f"Successfully created {output_filename} with {len(parsed_data['items'])} FAQ items!"
    )
    print(f"File location: {full_path}")


if __name__ == "__main__":
    generate_cs50_json()