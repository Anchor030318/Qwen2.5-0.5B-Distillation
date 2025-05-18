from dotenv import load_dotenv
load_dotenv()  # 加载.env中的所有配置

import os
import huggingface_hub

# 打印当前配置
print(f"使用的HF端点: {os.environ.get('HF_ENDPOINT', '未设置')}")
print(f"HF_HUB_BASE_URL: {os.environ.get('HF_HUB_BASE_URL', '未设置')}")

# 使用token直接登录(如果在.env中设置了HF_TOKEN)
if "HF_TOKEN" in os.environ and os.environ["HF_TOKEN"]:
    huggingface_hub.login(token=os.environ["HF_TOKEN"], add_to_git_credential=True)
    print("已使用环境变量中的token自动登录")
else:
    # 调用交互式登录
    huggingface_hub.login()