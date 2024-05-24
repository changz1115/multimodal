from huggingface_hub import HfApi, HfFolder

# 替换为你的 Hugging Face 用户令牌
token = "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"

# 保存令牌
HfFolder.save_token(token)

# 创建 HfApi 实例
api = HfApi()

# 验证登录是否成功
user_info = api.whoami()
print("Successfully logged in as:", user_info["name"])