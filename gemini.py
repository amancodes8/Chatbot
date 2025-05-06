from google import genai

client = genai.Client(api_key="AIzaSyDYOw-uy2EynM-uX6PYIg6xG9wGXJKjHvE")

response = client.models.generate_content(
    model="gemini-2.0-flash", contents="what's skin?
"
)
print(response.text)