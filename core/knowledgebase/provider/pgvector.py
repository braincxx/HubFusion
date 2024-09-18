from typing import List

import os
import asyncpg
import psycopg2

from core.knowledgebase.base import Vector, VectorDatabaseInterface, Data, Vector


class PgVectorInterface(VectorDatabaseInterface):
    
    def __init__(self, DB_CONFIG: dict):
        
        print(DB_CONFIG)
        self._DB_CONFIG = DB_CONFIG
        
        self.execute("CREATE EXTENSION IF NOT EXISTS vector")
    
    def connect(self) -> psycopg2.extensions.connection:
        try:
            connection = psycopg2.connect(**self._DB_CONFIG)
            return connection
        except Exception as e:
            print(f"Error connecting to database: {e}")
            
        
    def disconnect(self, connection: psycopg2.extensions.connection) -> None:
        try:
            connection.close()
        except Exception as e:
            print(f"Error disconnecting from database: {e}")
    
    def execute(self, query: str, params: tuple = None) -> None:
        """
        Execute a SQL query on the database.

        Args:
        - query (str): The SQL query to execute.
        - params (tuple, optional): The parameters to pass to the query.

        Returns:
        - None
        """
        connection = self.connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                connection.commit() 
                self.disconnect(connection)
                return True
        except Exception as e:
            print(f"Error executing query: {e}")
        finally:
            self.disconnect(connection)
        return False
    
    def fetch(self, query: str, params: tuple = None) -> List:
        
        connection = self.connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute(query, params)            
                result = cursor.fetchall()
                connection.commit()
                self.disconnect(connection)
                return result
        except Exception as e:
            print(f"Error executing query: {e}")
        finally:
            self.disconnect(connection)
        return []
    
    def insert(self, data: Data, vector: Vector) -> str:

        return self.execute('INSERT INTO documents ("text", embedding) VALUES (%s, %s)', 
                                (data, str(vector)))
        

    def search(self, vector: Vector, top_k: int = 10, modality: str | None = None) -> List[dict]:
    
        try:
            rows =  self.fetch("""
                                    SELECT "text"
                                    FROM documents
                                    ORDER BY embedding <-> %s
                                    LIMIT %s;
                                """, 
                                (str(vector), top_k))
        
            return [row[0] for row in rows]
        except Exception as e:
            return f"Error querying similar documents: {e}"

    
