import time
from requests import get, post
import sys
import requests
import json


def send_message(to_user, access_token):
    url = "https://api.weixin.qq.com/cgi-bin/message/template/send?access_token={}".format(access_token)
    data = {
        "touser": to_user,
        "template_id": "aSxKX7Llv8rGQng7OzvTD45XyAhzJfUFQZ9zcKNWSQA",
        "url": "http://weixin.qq.com/download",
        "topcolor": "#FF0000",
        "data": {
            }
        }

    headers = {
        'Content-Type': 'application/json',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36'
    }
    response = post(url, headers=headers, json=data).json()
    if response["errcode"] == 40037:
        print("推送消息失败，请检查模板id是否正确")
    elif response["errcode"] == 40036:
        print("推送消息失败，请检查模板id是否为空")
    elif response["errcode"] == 40003:
        print("推送消息失败，请检查微信号是否正确")
    elif response["errcode"] == 0:
        print("推送消息成功")
    else:
        print(response)


def get_access_token():
    # appId
    app_id = "wxf36fe652cc123dd3"
    # appSecret
    app_secret = "0b031cff0e9f3b32c846ca4458d539ae"
    post_url = ("https://api.weixin.qq.com/cgi-bin/token?grant_type=client_credential&appid={}&secret={}"
                .format(app_id, app_secret))
    try:
        access_token = get(post_url).json()['access_token']
    except KeyError:
        print("获取access_token失败，请检查app_id和app_secret是否正确")
        sys.exit(1)
    # print(access_token)
    return access_token


if __name__ == "__main__":

    # 获取accessToken
    accessToken = get_access_token()
    # 接收的用户
    user = "oqs5k6MGcmJ2rCxfhsqzzUk8JfXc"
    # 公众号推送消息
    send_message(user, accessToken)
