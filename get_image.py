from supabase import create_client, Client
from io import BytesIO
from PIL import Image, ImageOps

def image_downloader(bucket_name, bucket_id):

    url: str = "https://txzmfottvrwpxwfttemf.supabase.co/"
    key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR4em1mb3R0dnJ3cHh3ZnR0ZW1mIiwicm9sZSI6ImFub24iLCJpYXQiOjE2OTkwMDUxODAsImV4cCI6MjAxNDU4MTE4MH0.GcQBkFXhhMRkyRtMg9XpYYomA3nUaSyriYyLliKwqW0"

    supabase: Client = create_client(url, key)

    data = supabase.storage.from_(bucket_name).download(bucket_id)

    img = Image.open(BytesIO(data))

    img.save("./get_image/test.jpg")