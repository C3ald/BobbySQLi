import requests as rep
import re


class Website:
    """ tests for sql injection """

    def __init__(self, url, method="GET", json=False, param="id", cookie={}):
        self.url = url
        self.method = method
        self.json = json
        self.param = param
        self.req = None
        self.r = rep.Session()
        if cookie:
            self.r.cookies.set_cookie(cookie)

    def inject(self, current_payload):
        try:
            if self.json == True:
                if self.method == "GET":
                    self.req = self.r.get(
                        self.url, json={param: current_payload})
                if self.method == "POST":
                    self.req = self.r.post(
                        self.url, json={param: current_payload})
            else:
                if self.method == "GET":
                    self.req = self.r.get(
                        self.url, params={param: current_payload})
                else:
                    self.req = self.r.post(
                        self.url, data={param: current_payload})
            return 1
        except:
            self.req = None
            None

    def is_vulnerable(self) -> bool or float:
            """ checks to see if the site is vulnerable to SQL injection"""
            error_messages = ["Syntax error","Access denied for user","Incorrect syntax near","Incorrect syntax for","Incorrect syntax near","Table or view does not exist","Column not found"]
            code = self.req.status_code
            response = self.req.text
            if code < 400:
                    if response in error_messages:
                            return True
                    else:
                            return 0.5
            return False
            
            