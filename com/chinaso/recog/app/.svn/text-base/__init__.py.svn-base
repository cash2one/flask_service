# -*- coding:utf-8 -*-
from flask import Flask

from com.chinaso.recog.util.server import Util
from flask_sqlalchemy import SQLAlchemy

server = Util()


app = Flask(__name__)
db = SQLAlchemy(app)

from com.chinaso.recog.app import views

# if __name__=="__main__":
#     app.run()