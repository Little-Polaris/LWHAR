import smtplib
from email.mime.text import MIMEText
import os
import platform  # 用于判断操作系统类型
import time  # 用于延迟

def send_email(sender_email, sender_password, receiver_email, subject, message):
    """
    发送电子邮件.

    Args:
        sender_email: 发件人邮箱地址.
        sender_password: 发件人邮箱密码 (建议从环境变量或配置文件读取).
        receiver_email: 收件人邮箱地址.
        subject: 邮件主题.
        message: 邮件正文.
    """
    try:
        # 配置邮件服务器 (这里以 Gmail 为例，其他邮箱需要修改服务器地址和端口)
        smtp_server = "smtp.qq.com"  # Gmail SMTP 服务器
        smtp_port = 587  # Gmail SMTP 端口 (TLS)

        # 创建 MIMEText 对象
        msg = MIMEText(message)
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = receiver_email

        # 连接到 SMTP 服务器
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # 启用 TLS 加密
            server.login(sender_email, sender_password)  # 登录邮箱
            server.sendmail(sender_email, receiver_email, msg.as_string())  # 发送邮件

        print("邮件发送成功!")

    except Exception as e:
        print(f"邮件发送失败: {e}")


def shutdown_system():
    """
    关闭计算机.
    """
    try:
        os_name = platform.system()  # 获取操作系统名称

        if os_name == "Windows":
            pass
            # os.system("shutdown /s /t 1")  # Windows 关机命令 (1 秒后关机)
        elif os_name == "Linux" or os_name == "Darwin":  # Darwin 是 macOS
            os.system("shutdown -h now")  # Linux/macOS 关机命令
        else:
            print("不支持的操作系统，无法关机.")
        print("系统正在关机...")

    except Exception as e:
        print(f"关机失败: {e}")


# 从环境变量中读取邮箱信息 (推荐)
def after_finish(shutdown_immediately: bool = False):
    sender_email = '2375425475@qq.com'
    sender_password = 'uihskvqfkuebecaj'
    receiver_email = 'jiahaoqi0519@gmail.com'  # 替换为你的收件人邮箱

        # 发送邮件
    subject = "Python 脚本运行结束通知"
    log_times = os.listdir('./logs')
    latest_log = open(f'./logs/{sorted(log_times)[-1]}/log.txt', 'r')
    message = ''.join(latest_log.readlines())
    send_email(sender_email, sender_password, receiver_email, subject, message)
    time.sleep(60)
    if not shutdown_immediately:
        time.sleep(300)
    # 关机
    shutdown_system()
