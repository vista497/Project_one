import psycopg2
from psycopg2 import Error

err=True

class DateBase():
    """Обертка для работы с базой данных Екатерины"""
    def __init__(self) -> None:
        pass

    def session(self,param, request):
        """Принимает запрос (строкой)"""
        try:
            # Подключение к существующей базе данных
            self.connection = psycopg2.connect(user="postgres",
                                        password="postgres",
                                        host="127.0.0.1",
                                        port="5432",
                                        database="ekaterinadb")
            # Курсор для выполнения операций с базой данных
            self.sess = self.connection.cursor()

            #собсна выполнение запроса
            
            if param=="select":
                self.sess.execute(request)
                response = self.sess.fetchone()
                return response
            if param=="insert":
                self.sess.execute(request)
                self.connection.commit()
            

        except (Exception, Error) as error:
            print("Ошибка при работе с PostgreSQL", error)
            return err

        finally:
            if self.connection:
                self.sess.close()
                self.connection.close()
                print("Соединение с PostgreSQL закрыто")
    
    def open(self):
        # Подключение к существующей базе данных
        connection = psycopg2.connect(user="postgres",
                                    password="postgres",
                                    host="127.0.0.1",
                                    port="5432",
                                    database="ekaterinadb")
        # Курсор для выполнения операций с базой данных
        self.connect = connection.cursor()
        return self.connect

    def close(self):
        if self.connection:
                self.sess.close()
                self.connection.close()
                print("Соединение с PostgreSQL закрыто")



