import session

SELECT="select"
INSERT="insert"

class Repository():
    def __init__(self):
        self.db=session.DateBase()

    def urlFromName(self, value):
        """Возвращает URL, принимает имя URL"""
        query="select url from web where name="+value
        resp=self.webStr=self.db.session(SELECT, query)
        return resp

    def insertUrl(self, name, url):
        """Добавляет новый URL c его именем"""
        query="insert into web (name, url) values ('"+name+ "', '"+url+"')"
        self.webStr=self.db.session(INSERT, query)
    
    def tgReg(self, firstName, lastName, age, tg_id, status):
        """Добавляет нового пользователя из чат бота"""

        query="insert into people_tg (first_name, last_name, age, tg_id) values ('"+firstName+"', '"+lastName+"','"+str(age)+"','"+str(tg_id)+"','"+status+"')"
        resp=self.db.session(INSERT, query)

    def getPersonById(self, tg_id):
        """Возвращает имя фамилию возраст и айди""" 
        sess=self.db.open()
        query="select (first_name,last_name, age, tg_id) from person_tg where tg_id="+str(tg_id)
        sess.execute(query)
        response = sess.fetchone()
        self.db.close()
        first_name=response[0]
        last_name=response[1]
        age=response[2]
        tg_id=response[3]
        return first_name, last_name, age, tg_id
        
    
 
