import smtplib
from email.mime.text import MIMEText

def send_email(to, subject, body):
    try:
        # Email details
        sender_email = 'assmaha787@gmail.com'
        app_password = 'wpzdqoaokprwpoiy'  # Replace with the app password

        # Create the email content
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = to

        # Use Gmail's SMTP server
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, app_password)  # Use the app password here
            server.sendmail(sender_email, to, msg.as_string())

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False
