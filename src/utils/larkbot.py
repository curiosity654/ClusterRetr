# larkbot.py

import requests

class LarkBot:
    def __init__(self, url: str) -> None:
        self.WEBHOOK_URL = url

    def send(self, content: str) -> None:

        params = {
            "msg_type": "text",
            "content": {"text": content},
        }
        resp = requests.post(url=self.WEBHOOK_URL, json=params)
        resp.raise_for_status()
        result = resp.json()
        if result.get("code") and result["code"] != 0:
            print(result["msg"])
            return
        print("消息发送成功")

def main():
    WEBHOOK_URL = "https://open.feishu.cn/open-apis/bot/v2/hook/03fdc834-de4b-41a9-8d15-7c8410d44915"
    bot = LarkBot(url=WEBHOOK_URL)
    bot.send(content="test")

if __name__ == '__main__':
    main()