import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import traceback

def send_email(subject, message, recipient_emails):
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    sender_email = 'osamaalaa.career@gmail.com'
    sender_password = 'o.osama1245*'  # Use app-specific password if 2-factor authentication is enabled

    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = ', '.join(recipient_emails)
        msg['Subject'] = subject
        msg.attach(MIMEText(message, 'plain'))

        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_emails, msg.as_string())
        server.quit()

        print("Email sent successfully!")
    except Exception as e:
        print("An error occurred while sending the email:")
        print(traceback.format_exc())



df = pd.read_csv("D:\handson-ml2\pipelines and ready functions\Emails.csv")

subject = 'Looking For Opportunity'
message = 'Hello there,I am Osama, a computer science graduate from Cairo University. I was wondering if, by any chance, you could help me find training or a part-time opportunity in data and Machine Learning. To gain practical experience and exploit my time to provide real benefits. Any chance would be really appreciated, Thanks in advance for your time. Best regards, Osama'

# Extract email addresses from the DataFrame column
recipient_emails = df['Email'].dropna().tolist()

# Send emails to the extracted email addresses
send_email(subject, message, recipient_emails)
