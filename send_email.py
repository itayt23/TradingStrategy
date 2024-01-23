from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
from datetime import datetime
import smtplib
import os
import traceback

APP_NAME = os.path.basename(__file__)
# LOGGER_NAME = os.path.basename(__file__)
PROJECT_NAME = os.path.basename(__file__)
load_dotenv()


sender_email = os.environ.get('GMAIL_SENDER')
sender_password = os.environ.get('GMAIL_PASSWORD')
receiver_emails = ["testalmog1@gmail.com","itayhcm@gmail.com"]
receiver_my_emails = ["testalmog1@gmail.com"]
cc_recipients = []  
sender_name = "Almog Tal"
developer_team = "Horizon Developer Team"

LP1_ACCOUNT = os.getenv('LP1_ACCOUNT')
LP2_ACCOUNT = os.getenv('LP2_ACCOUNT')

names = {LP1_ACCOUNT:'LP1',LP2_ACCOUNT:'LP2'}

def send_mail(account,df_to_alarm):
    if df_to_alarm.empty:return

    system_date = datetime.now().strftime("%d.%m.%y")
    subject = f"HIGH RISK BPS/BCS OPTIONS STATUS - {system_date}"

    start_massage = """<head>Below is a report of recommendations</head>"""
    end_massage = f"""<p>Please reply to this email if you believe there is a problem in it.</p>
                <p>Best Regards,<br>
                {sender_name},<br>
                {developer_team}</p>"""
    
    added_somthing = False
    list_massages = []
    list_massages.append(start_massage)
    
    
    list_massages.append(f"""<h1>The following data is relevant to {account} {names[account]} </h1>""")
    list_massages.append(df_to_alarm.to_html())
    list_massages.append(end_massage)

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = ", ".join(receiver_emails)
    message["Subject"] = subject
    message["Cc"] = ", ".join(cc_recipients)

    for item in list_massages:
        message.attach(MIMEText(item, 'html'))

    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    if added_somthing:    
     
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)

        server.sendmail(sender_email, receiver_emails+cc_recipients, message.as_string())

        server.quit()
        # logger.log_struct({"message": f"{APP_NAME}: Sent All Emails successfully!","severity": "INFO"})


def send_me_email(text):

    system_date = datetime.now().strftime("%d.%m.%y")
    subject = f"Cloud Script Alert - {system_date} - TradingStrategy"

    end_massage = f"""Please reply to this email if you believe there is a problem in it.
Best Regards,
{sender_name},
{developer_team}"""

    message_with_subject = f"Subject: {subject}\n\n{text}\n\n\n{end_massage}"
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()
    server.login(sender_email, sender_password)

    server.sendmail(sender_email, receiver_my_emails, message_with_subject)

    server.quit()
