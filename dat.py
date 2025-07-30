
#AIzaSyAOV0pCcNbs7QpEZUWMLENIZLK-Kfuu6yY

import os
import time
from google import genai
from google.genai import types

API Key Gemini

API_KEY = "AIzaSyDokaO_-S2t1S9J2KeCQk8iRD7-pcmDkSc"
genai_client = genai.Client(api_key=API_KEY)

Cấu hình

NUM_NEW_PARAGRAPHS = 5
DATA_FILE = "data.txt"

def load_old_data(limit: int = 300) -> str:
if not os.path.exists(DATA_FILE):
return ""
with open(DATA_FILE, "r", encoding="utf-8") as f:
lines = f.readlines()
return "".join(lines[-limit:])

def save_new_data(new_data: str):
with open(DATA_FILE, "a", encoding="utf-8") as f:
f.write(new_data.strip() + "\n")

def extract_code_block(text: str) -> str:

Lấy nội dung trong ...

if "" in text:   parts = text.split("")
for part in parts:
lines = part.strip().splitlines()
if len(lines) >= 5:  # có vẻ là block dữ liệu thực
return "\n".join(lines).strip()
return text.strip()

def generate_batch():
old_data = load_old_data()
prompt = f"""Bạn hãy sinh thêm {NUM_NEW_PARAGRAPHS} MB đoạn văn bản ngắn bằng tiếng Việt để huấn luyện pretrain cho một mô hình ngôn ngữ, chất lượng cực mạnh, ít nhưng pretrain vẫn ổn.
80% dạng mô tả lời chào, đối thoại, cách trả lời câu hỏi, tình huống xã hội,... giúp model hình thành phản xạ.
→ Ví dụ:

Khi được hỏi "input?", bạn có thể trả lời "output".

20% fact để giữ cân bằng kiến thức nền:

Trái Đất quay quanh Mặt Trời mất khoảng 365 ngày.
LƯU Ý : CHỈ GHI DATA KHÔNG GHI BẤT KỲ CÁI GÌ KHÁC
MẪU :
.....
.....
.....

Yêu cầu:

Mỗi đoạn tối đa 120 từ

Mỗi dòng là 1 đoạn văn không cần đánh số

Các đoạn liên quan giao tiếp đời thường, story mạng xã hội, giới thiệu

Không trùng lặp với đoạn đã có

Đơn giản, rõ ràng, không viết kiểu hội thoại
Đoạn đã có:
{old_data.strip()}
"""

contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]

try:
raw = ""
for chunk in genai_client.models.generate_content_stream(
model="gemini-2.5-pro",
contents=contents,
config=types.GenerateContentConfig(),
):
raw += chunk.text

clean_data = extract_code_block(raw)
print(clean_data)
save_new_data(clean_data)

except Exception as e:
print("Lỗi khi gọi Gemini:", e)

if name == "main":
while True:
print("\n=== ĐANG SINH DỮ LIỆU ===")
generate_batch()
print("\n=== ĐỢI 10 GIÂY TIẾP TỤC ===")
time.sleep(10)

