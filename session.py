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
            connection = psycopg2.connect(user="postgres",
                                        password="123",
                                        host="127.0.0.1",
                                        port="5432",
                                        database="dbekaterina")
            # Курсор для выполнения операций с базой данных
            self.sess = connection.cursor()

            #собсна выполнение запроса
            
            if param=="select":
                self.sess.execute(request)
                response = self.sess.fetchone()
            if param=="insert":
                self.sess.execute(request)
                connection.commit()
            return response

        except (Exception, Error) as error:
            print("Ошибка при работе с PostgreSQL", error)
            return err

        finally:
            if connection:
                self.sess.close()
                connection.close()
                print("Соединение с PostgreSQL закрыто")



