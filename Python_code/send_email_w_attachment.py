#  ##########################################################################
# the following method is used to send email from a Gmail account with
# subject, body, and file attachment
# ###########################################################################

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# USE THIS ACCOUNT TO SEND EMAILS:
# email: imes.data@gmail.com
# password: imes.data2019

def send_email(to_address,
               subject,
               body,
               attachment_file=None,
               from_address='imes.data@gmail.com',
               password_of_sender='imes.data2019'):
    # send an email wtith attachment using a gmail account as the sender
    # instance of MIMEMultipart
    msg = MIMEMultipart()
    msg['From'] = from_address
    msg['To'] = to_address
    msg['Subject'] = subject
    # attach the body with the msg instance
    msg.attach(MIMEText(body, 'plain'))
    if attachment_file:
        # open the file to be sent
        attachment = open(attachment_file, "rb")
        # instance of MIMEBase and named as p
        p = MIMEBase('application', 'octet-stream')
        # To change the payload into encoded form
        p.set_payload((attachment).read())
        # encode into base64
        encoders.encode_base64(p)
        p.add_header('Content-Disposition',
                    'attachment; filename= %s' % attachment_file)
        # attach the instance 'p' to instance 'msg'
        msg.attach(p)

    # creates SMTP session
    s = smtplib.SMTP('smtp.gmail.com', 587, timeout=30)
    # start TLS for security
    s.starttls()
    # Authentication
    s.login(from_address, password_of_sender)
    # Converts the Multipart msg into a string
    text = msg.as_string()
    # sending the mail
    s.sendmail(from_address, to_address, text)
    # terminating the session
    s.quit()


send_email('ericmuckley@gmail.com', 'SUB HI', 'BODY   ...',
           attachment_file=r'C:\Users\a6q\qcm_fit_peaks.py')