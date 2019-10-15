# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 16:44:49 2018

@author: a6q
"""

# Retrieved from http://solvingmytechworld.blogspot.com/2013/01/send-email-through-gmail-running-script.html

# Import smtplib for the actual sending function
import smtplib
 # For guessing MIME type
from email.mime.text import  MIMEText
from email.mime.multipart import MIMEMultipart
# Import the email modules we'll need
import email
import email.mime.application
 
# Create a text/plain message
msg = MIMEMultipart()
msg['Subject'] = 'subject_you_prefer'
msg['From'] = 'from@gmail.com'
msg['To'] = 'receivers_mail@gmail.com'

 
# The main body is just another attachment
body = MIMEText("""Here you can write as many things as you want!""")
msg.attach(body)
 

directory = r'C:\Users\a6q\Desktop\lab on a QCM\images\2018-10-05_pedotpss.JPG'
# Split de directory into fields separated by / to substract filename
spl_dir=directory.split('/')
# We attach the name of the file to filename by taking the last
# position of the fragmented string, which is, indeed, the name
# of the file we've selected
filename='C:\\Users\\a6q\\exp_data\\2018-11-19_pss_bodelist.csv' #spl_dir[len(spl_dir)-1]
 
# We'll do the same but this time to extract the file format (pdf, epub, docx...)
spl_type=directory.split('.')
filetype=spl_type[len(spl_type)-1]
 
fp=open(directory,'rb')
att = email.mime.application.MIMEApplication(fp.read(),_subtype=filetype)
fp.close()
att.add_header('Content-Disposition','attachment',filename=filename)
msg.attach(att)
 
# send via Gmail server
# NOTE: my ISP, Centurylink, seems to be automatically rewriting
# port 25 packets to be port 587 and it is trashing port 587 packets.
# So, I use the default port 25, but I authenticate.
s = smtplib.SMTP('smtp.gmail.com:587')
s.starttls()
s.login('ericmuckley@gmail.com','PASSWORD')
s.sendmail('ericmuckley@gmail.com','muckleyes@ornl.gov', msg.as_string())
s.quit()
