# -*- coding: utf-8 -*-
"""

This script controls functions of an SQLite database.


To examine database in Anaconda prompt, use
sqlite3 c:\\users\\a6q\\sample_db.db
.header on
.mode column
SELECT * FROM samples;
SELECT * FROM history;


.quit



Created on Mon Nov  4 18:17:55 2019
@author: ericmuckley@gmail.com
"""

import sqlite3
 
 

def db_connect(db_file):
    """Create a database connection to a SQLite database.
    If the database does not exist, it is created."""
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except sqlite3.Error as e:
        print(e)
    return conn
    

    
def create_table(conn, create_table_sql):
    """Create a table from the create_table_sql statement using a 
    connection object and the string SQL CREATE TABLE statement."""
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except sqlite3.Error as e:
        print(e)   
   
    
def create_sample(conn, sample):
    """Create a new sample for insertion into the samples table.
    Returns the sample id."""
    sql = '''INSERT INTO samples(name, creator, composition,
                                fabrication_date, fabrication_method, notes)
              VALUES(?,?,?,?,?,?)'''
    cur = conn.cursor()
    cur.execute(sql, sample)
    return cur.lastrowid   


def create_event(conn, event):
    """Create a new event for insertion into the history table.
    Returns the event id."""
    sql = '''INSERT INTO history(sample_id, date, event, notes)
              VALUES(?,?,?,?)'''
    cur = conn.cursor()
    cur.execute(sql, event)
    return cur.lastrowid   


def print_table(conn, table_name):
    """Query all rows in the samples table."""
    cur = conn.cursor()
    cur.execute('SELECT * FROM ' + str(table_name))
    rows = cur.fetchall()
    for row in rows:
        print(row)


def select_sample_by_creator(conn, creator):
    """Query events by sample id."""
    cur = conn.cursor()
    cur.execute("SELECT * FROM samples WHERE creator=?", (creator,))
    rows = cur.fetchall()
    for row in rows:
        print(row)


def edit_event(conn, event):
    """Update event of a sample."
    Event should be in the form of a tuple:
    ('date', 'event_name', 'notes', sample_id)
    """
    sql = '''UPDATE history
              SET date = ? ,
                  event = ? ,
                  notes = ?
              WHERE id = ?'''
    cur = conn.cursor()
    cur.execute(sql, event)
    conn.commit()


def edit_sample(conn, sample):
    """Update sample information."""
    sql = '''UPDATE samples
              SET name = ? ,
                  creator = ? ,
                  composition = ? ,
                  fabrication_date = ? ,
                  fabrication_method = ? ,
                  notes = ?
              WHERE id = ?'''
    cur = conn.cursor()
    cur.execute(sql, sample)
    conn.commit()




# SQL commands for creating tables in the database
sql_create_samples_table = '''CREATE TABLE IF NOT EXISTS samples (
                                    id integer PRIMARY KEY,
                                    name text NOT NULL,
                                    creator text NOT NULL,
                                    composition text NOT NULL,
                                    fabrication_date text NOT NULL,
                                    fabrication_method text NOT NULL,
                                    notes text
                                );'''

sql_create_history_table = '''CREATE TABLE IF NOT EXISTS history (
                                    id integer PRIMARY KEY,
                                    sample_id integer NOT NULL,
                                    date text NOT NULL,
                                    event text NOT NULL,
                                    notes text,
                                    FOREIGN KEY (sample_id) REFERENCES samples (id)
                                );'''

sample1 = ('em-sampleX', 'eric', 'PEDOT:PSS 1:5 ratio', '2019-11-05',
           'spin-coating at 50 degrees C',
           'used old PEDOT:PSS from Bin Hu')

sample2 = ('Y', 'ericM', 'PEDOT:PSS', '2018-11-05',
           'dropcast',
           '')


sample3 = ('NEWNAME', 'joe', 'PEDOT:PSS 1:5 ratio', '2019-11-05',
           'spin-coating at 50 degrees C',
           'used old PEDOT:PSS from Bin Hu')




if __name__ == '__main__':
    
    # connect to database
    db_file = r'C:\Users\a6q\sample_db.db'
    conn = db_connect(db_file)
    
    
    if conn:
        
        # create tables
        create_table(conn, sql_create_samples_table)
        create_table(conn, sql_create_history_table)
        
        # add sample
        sample_id = create_sample(conn, sample1)
        # add event history
        event1 = (sample_id, '2019-10-21', 'measured CV', 'looked good')
        event2 = (sample_id, '2019-11-01', 'measured imedpedance', 'sample shorted')
        create_event(conn, event1)
        create_event(conn, event2)


        # add sample
        sample_id = create_sample(conn, sample2)
        # add event history
        event1 = (sample_id, '2019-11-21', 'ellipsometry', '')
        event2 = (sample_id, '2019-12-01', 'RH experiments', '')
        create_event(conn, event1)
        create_event(conn, event2)


        edit_event(conn, ('2001-06-06', 'UPDATED', 'NO NOTES', sample_id))
        
        edit_sample(conn, sample3 + (sample_id,))
        
        

        print_table(conn, 'samples')
        print_table(conn, 'history')

        select_sample_by_creator(conn, 'eric')

        
        
        




    conn.close()




class table(object):
    def __init__(self, name):
        self.name = name
    def add_str(self, string):
        self.name = self.name+string

