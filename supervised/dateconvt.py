import datetime

def datestr2num(s):
    return datetime.datetime.strptime(s, "%Y-%m-%d").date().weekday() 
