import session

SELECT="select"
INSERT="insert"

class Repository():
    def __init__(self):
        self.db=session.DateBase()

    def urlFromName(self, value):
        """Возвращает URL, принимает имя URL"""
        str="select url from web where name="+value
        resp=self.webStr=self.db.session(SELECT, str)
        return resp

    def insertUrl(self, name, url):
        """Добавляет новый URL c его именем"""
        str="insert into web (name, url) values ('"+name+ "', '"+url+"')"
        self.webStr=self.db.session(INSERT, str)


