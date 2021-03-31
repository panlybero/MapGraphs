import json

def read_config(fname):
    f = open(fname,'r')
    config = json.load(f)
    f.close()
    return config