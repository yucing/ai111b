# 程式碼
* 以下為用 ChatGPT 所產生的程式碼，無進行修改
## 智慧推薦系統
```py
import openai

# 設置 OpenAI API 密鑰
openai.api_key = '你的API密鑰'

# 定義用戶的輸入
user_input = "我正在尋找一本關於科幻的書籍。"

# 使用 OpenAI GPT-3 模型生成推薦文本
response = openai.Completion.create(
  engine="davinci",
  prompt=user_input,
  max_tokens=100,
  n=5,  # 生成多個候選推薦文本
  stop=None,
  temperature=0.7
)

# 提取推薦文本
recommendations = [choice['text'].strip() for choice in response['choices']]

# 輸出推薦結果
print("推薦的文本:")
for index, recommendation in enumerate(recommendations):
    print(f"推薦 {index+1}: {recommendation}")

```

# 執行結果
* 因為我的額度已經到了QQ沒辦法用

![](https://github.com/yucing/ai111b/blob/main/picture/1.jpg)