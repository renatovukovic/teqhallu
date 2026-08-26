# coding=utf-8
#
# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Renato Vukovic (renato.vukovic@hhu.de)
#
# This code was generated with the help of AI writing assistants
# including GitHub Copilot, ChatGPT, Bing Chat.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import sqlite3
import re
import numpy as np
import random
import sqlglot


def extract_sql_from_response(response: str) -> list[str]:
    """
    Extract the SQL queries from the response and remove newlines from them.

    Input:
        response: str: the response from the model that contains SQL queries

    Output:
        list[str]: the SQL queries extracted from the response with no newlines
    """
    # Capture all SQL queries (multi-line enabled)
    sql_queries = re.findall(r"```sql(.*?)```", response, re.DOTALL)

    return sql_queries

def execute_query(database_name: str, query: str) -> str or list:
    """
    Executes one SQL query on a given SQLite database.
    
    Parameters:
        database_name (str): The name of the SQLite database file.
        query (str): The SQL query to execute.
        
    Returns:
        str or list: The result of the query (for SELECT) or a confirmation message (for INSERT/UPDATE).
    """
    try:
        # Connect to the specified SQLite database (creates it if it doesn't exist)
        connection = sqlite3.connect(database_name)
        cursor = connection.cursor()
        
        # Execute the query
        cursor.execute(query)
        
        # Commit if the query modifies data (e.g., INSERT, UPDATE, DELETE)
        if query.strip().upper().startswith(("INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER")):
            connection.commit()
            result = "Update completed and saved successfully."
        else:
            # Fetch data if it's a SELECT query
            result = cursor.fetchall()
        
        # Close the connection
        connection.close()
        
        # Return the result
        return result
    
    except sqlite3.Error as e:
        # Catch any SQL errors and return the error message
        return f"An error occurred: {e}"

# Example usage
# database_name = 'employees.db'
# query = "SELECT * FROM employees WHERE salary > 50000;"  # Example query; replace with any SQL command
# result = execute_query(database_name, query)
# print(result)


def execute_queries(database_name: str, queries: str) -> str or list:
    """
    Executes multiple SQL queries in a straing on a given SQLite database with the execute script function from sqlite. Can only return the result of a SELECT query if it was the last query.
    
    Parameters:
        database_name (str): The name of the SQLite database file.
        queries (str): A string containing multiple SQL queries, separated by semicolons.
        
    Returns:
        str or list: The result of the last SELECT query, or a confirmation message for updates. 
                     Returns error message if any query fails.
    """
    try:
        # Connect to the specified SQLite database (creates it if it doesn't exist)
        connection = sqlite3.connect(database_name)
        cursor = connection.cursor()
        
        # Use executescript to run multiple queries in one call
        cursor.executescript(queries)
        
        # If the last query is a SELECT, fetch and return its results
        # We'll split the queries and check if the last one is a SELECT
        last_query = queries.strip().split(';')[-2].strip()  # Get the last SQL query without trailing semicolons
        
        if last_query.upper().startswith("SELECT"):
            cursor.execute(last_query)
            result = cursor.fetchall()
        else:
            # Commit if there were any INSERT, UPDATE, DELETE, etc., statements
            connection.commit()
            result = "Update completed and saved successfully."

        # Close the connection
        connection.close()
        
        # Return the result
        return result
    
    except sqlite3.Error as e:
        # Catch any SQL errors and return the error message
        return f"An error occurred: {e}"

# Example usage
# database_name = 'employees.db'
# queries = """
# CREATE TABLE IF NOT EXISTS employees (
#     id INTEGER PRIMARY KEY,
#     name TEXT,
#     position TEXT,
#     salary REAL
# );
# INSERT INTO employees (name, position, salary) VALUES ('Alice', 'Engineer', 60000);
# INSERT INTO employees (name, position, salary) VALUES ('Bob', 'Manager', 80000);
# SELECT * FROM employees;
# """

# result = execute_queries(database_name, queries)
# print(result)


def execute_multiple_queries(database_name: str, queries: str):
    """
    Executes multiple SQL queries on a given SQLite database, ignoring comments and 
    correcting escaped single quotes.
    
    Parameters:
        database_name (str): The name of the SQLite database file.
        queries (str): A string containing multiple SQL queries, separated by semicolons.
                       Can contain SQL comments (line and block comments).
        
    Returns:
        list of tuples: A list where each tuple contains a query and its result or error message.
                        - For successful SELECT queries: (query, result)
                        - For non-SELECT updates: (query, "Update completed")
                        - For errors: (query, error message)
    """
    try:
        # Connect to the specified SQLite database
        connection = sqlite3.connect(database_name)
        cursor = connection.cursor()
        
        # Remove comments and fix single quotes in the SQL string
        def preprocess_sql(sql):
            # Remove block comments (/* ... */)
            sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
            # Remove single-line comments (--) and any whitespace after them
            sql = re.sub(r'--.*', '', sql)
            # Replace escaped single quotes (\') with SQLite-compatible ('' for single quote)
            sql = sql.replace("\\'", "''")
            return sql
        
        # Preprocess the queries string to remove comments and fix single quotes
        queries = preprocess_sql(queries)
        
        # Split the cleaned string into individual SQL statements
        query_list = [query.strip() for query in queries.strip().split(';') if query.strip()]
        
        # Initialize an empty list to store (query, result) for each query
        results = []
        
        # Execute each query individually
        for query in query_list:
            try:
                if query.upper().startswith("SELECT"):
                    # Execute SELECT query and store the (query, result) tuple
                    cursor.execute(query)
                    results.append((query, cursor.fetchall()))
                else:
                    # Execute non-SELECT query, commit changes, and store update confirmation
                    cursor.execute(query)
                    connection.commit()
                    results.append((query, "Update completed"))
            except sqlite3.Error as e:
                # Append the error message if an error occurs with the current query
                results.append((query, f"Error: {e}"))
        
        # Close the connection
        connection.close()
        
        # Return the list of (query, result or error message) tuples
        return results
    
    except sqlite3.Error as e:
        # Catch any SQL errors during connection setup or closure and return the error message
        return f"An error occurred with the database connection: {e}"

# # Example usage
# database_name = 'employees.db'
# queries = """
# -- This is a line comment
# CREATE TABLE IF NOT EXISTS employees (
#     id INTEGER PRIMARY KEY,
#     name TEXT,
#     position TEXT,
#     salary REAL
# );
# /* Insert some test data */
# INSERT INTO employees (name, position, salary) VALUES ('Alice', 'Engineer', 60000);
# INSERT INTO employees (name, position, salary) VALUES ('Bob', 'Manager', 80000);
# INSERT INTO employees (name, position, salary) VALUES ('Saint John''s', 'Teacher', 55000);
# SELECT * FROM employees; -- Select all employees
# """

# result = execute_multiple_queries(database_name, queries)
# for query, output in result:
#     print(f"Query: {query}\nOutput: {output}\n")




def execute_multiple_queries_with_errors(database_name: str, 
                                         queries: str,
                                         do_not_execute_update_queries: bool = False) -> list[tuple]:
    """
    Executes multiple SQL queries on a given SQLite database, ignoring comments, 
    correcting escaped single quotes, and attempting to fix common errors.
    
    Parameters:
        database_name (str): The name of the SQLite database file.
        queries (str): A string containing multiple SQL queries, separated by semicolons.
                       Can contain SQL comments (line and block comments).
        do_not_execute_update_queries (bool): Whether to execute non-SELECT queries (e.g., INSERT, UPDATE, DELETE) or just return them as is, since they should only be generated in the last step, in the prior steps they will not be executed, since they are falsely generated, there should only be select and pragma queries in these steps.
        
    Returns:
        list of tuples: A list where each tuple contains a query and its result or error message.
                        - For successful SELECT queries: (query, result)
                        - For non-SELECT updates: (query, "Update completed")
                        - For errors: (query, error message)
    """
    try:
        # Connect to the specified SQLite database
        connection = sqlite3.connect(database_name)
        cursor = connection.cursor()
        
        # Remove comments and fix single quotes in the SQL string
        def preprocess_sql(sql):
            # Remove block comments (/* ... */)
            sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
            # Remove single-line comments (--) and any whitespace after them
            sql = re.sub(r'--.*', '', sql)
            # Replace escaped single quotes (\') with SQLite-compatible ('' for single quote)
            sql = sql.replace("\\'", "''")
            return sql
        
        # Preprocess the queries string to remove comments and fix single quotes
        queries = preprocess_sql(queries)
        
        # Split the cleaned string into individual SQL statements
        query_list = [query.strip() for query in queries.strip().split(';') if query.strip()]
        
        # Initialize an empty list to store (query, result) for each query
        results = []
        
        # Execute each query individually
        for query in query_list:
            try:
                if query.upper().startswith("SELECT") or query.upper().startswith("PRAGMA"):
                    # Execute SELECT query and store the (query, result) tuple
                    cursor.execute(query)
                    query_result = cursor.fetchall()
                    results.append((query, query_result))
                else:
                    # Execute non-SELECT query, commit changes, and store update confirmation
                    cursor.execute(query)
                    if do_not_execute_update_queries:
                        results.append((query, "Update query not executed yet."))
                    else:
                        connection.commit()
                        results.append((query, "Update completed"))
            except sqlite3.Error as e:
                # Handle common errors and attempt to fix the query
                fixed_query = handle_sqlite_errors(query, e)
                
                if fixed_query != query:
                    # If the query was modified, re-execute the fixed query
                    try:
                        cursor.execute(fixed_query)
                        if do_not_execute_update_queries:
                            results.append((query, "fixed Update query not executed yet."))
                        else:
                            connection.commit()
                            results.append((query, "Update completed with fixed query above"))
                    except sqlite3.Error as fixed_error:
                        results.append((fixed_query, f"Error after fixing: {fixed_error}"))
                else:
                    results.append((query, f"Error: {e}"))
        
        # Close the connection
        connection.close()
        
        # Return the list of (query, result or error message) tuples
        return results
    
    except sqlite3.Error as e:
        # Catch any SQL errors during connection setup or closure and return the error message
        return f"An error occurred with the database connection: {e}"

def handle_sqlite_errors(query, error):
    """
    Attempts to fix common SQLite errors in the query by removing the problematic part.
    It only executes the part of the query before the error.

    Parameters:
        query (str): The SQL query that caused the error.
        error (sqlite3.Error): The error object raised during the query execution.

    Returns:
        str: The fixed query (if applicable) or the original query if no fix is applied.
    """
    # Example: If the query contains 'ON DUPLICATE KEY UPDATE', remove that part
    if "ON DUPLICATE KEY UPDATE" in query:
        # Strip everything after 'ON DUPLICATE KEY UPDATE'
        fixed_query = query.split("ON DUPLICATE KEY UPDATE")[0]
        return fixed_query
    
    # Example: If ON CONFLICT is used improperly, remove that part
    if "ON CONFLICT" in query:
        # Strip everything after 'ON CONFLICT'
        fixed_query = query.split("ON CONFLICT")[0]
        return fixed_query
    
    if "AUTO_INCREMENT" in query: #remove it as it is not supported in SQLite
        fixed_query = query.replace("AUTO_INCREMENT", "")
        return fixed_query
    
    # Add more error handling cases as needed

    # If no fix is applied, return the original query
    return query


def get_entire_database_structure(database_name: str) -> dict:
    """
    Fetches the structure of the entire SQLite database as a dictionary, where:
    - Each table is a top-level key.
    - Each column within a table is a second-level key.
    - The values are sets containing all unique non-None values for each column.
    - Columns with None values will exist but will be empty if all values are None.

    Parameters:
        database_name (str): The SQLite database file name.

    Returns:
        dict: A dictionary representing the structure of the entire database.
    """

    # Initialize the dictionary to store the entire database structure
    database_structure = {}
    
    # Connect to the SQLite database
    connection = sqlite3.connect(database_name)
    cursor = connection.cursor()

    # Get all table names in the database
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    

    for table in tables:
        table_name = table[0].lower()
        # Initialize a dictionary for each table
        database_structure[table_name] = {}

        # Get all column names for the current table
        try:
            cursor.execute(f'PRAGMA table_info("{table_name}");')
            columns = cursor.fetchall()
        except sqlite3.Error as e:
            # Handle any errors that occur during the database operations
            print(f'Error accessing the database: {e}\nQuery was PRAGMA table_info("{table_name}");')
            continue
        
        # For each column, initialize a set to store its unique non-None values
        for column in columns:
            column_name = column[1].lower()
            database_structure[table_name][column_name] = set()

        # Now fetch all rows for the current table
        try:
            cursor.execute(f'SELECT * FROM "{table_name}";')
            rows = cursor.fetchall()
        except sqlite3.Error as e:
            # Handle any errors that occur during the database operations
            print(f'Error accessing the database: {e}\nQuery was SELECT * FROM "{table_name}";')
            continue

        # Populate the sets with unique non-None values from the rows
        for row in rows:
            for idx, value in enumerate(row):
                column_name = columns[idx][1].lower()
                if value is not None:  # Exclude None values
                    value = str(value)
                    value = value.lower()
                    database_structure[table_name][column_name].add(value)

        # Ensure that columns with only None values are empty sets
        for column in columns:
            column_name = column[1].lower()
            if not database_structure[table_name][column_name]:
                # Leave the column as an empty set if it contains no non-None values
                database_structure[table_name][column_name] = set()

    # Close the connection
    connection.close()

    return database_structure

    # except sqlite3.Error as e:
    #     # Handle any errors that occur during the database operations
    #     print(f"Error accessing the database: {e}")
    #     connection.close()
    #     return database_structure



###################### functions for extracting table names, column names and values from SQL queries ######################
def extract_table_names(sql_query, dialect="sqlite"):
    """
    Extracts table names from an SQL query using sqlglot, with optional dialect specification.
    Handles PRAGMA, CREATE, INSERT, UPDATE, ALTER, and other common SQL queries.

    Parameters:
        sql_query (str): The SQL query string.
        dialect (str): The SQL dialect to use for parsing (default is "sqlite").

    Returns:
        list: A list of table names found in the query.
    """
    try:
        # First, check for PRAGMA queries and extract the table name from them
        pragma_match = re.match(r"PRAGMA\s+table_info\((\w+)\)", sql_query.strip(), re.IGNORECASE)
        if pragma_match:
            return [pragma_match.group(1)]
        
        # Check for CREATE TABLE queries and extract the table name
        create_match = re.match(r"CREATE\s+TABLE\s+(\w+)", sql_query.strip(), re.IGNORECASE)
        if create_match:
            return [create_match.group(1)]
        
        # Check for INSERT INTO queries and extract the table name
        insert_match = re.match(r"INSERT\s+INTO\s+(\w+)", sql_query.strip(), re.IGNORECASE)
        if insert_match:
            return [insert_match.group(1)]
        
        # Check for UPDATE queries and extract the table name
        update_match = re.match(r"UPDATE\s+(\w+)", sql_query.strip(), re.IGNORECASE)
        if update_match:
            return [update_match.group(1)]
        
        # Check for ALTER TABLE queries and extract the table name
        alter_match = re.match(r"ALTER\s+TABLE\s+(\w+)", sql_query.strip(), re.IGNORECASE)
        if alter_match:
            return [alter_match.group(1)]
        
        # Otherwise, parse the SQL query using sqlglot for general SQL queries
        parsed = sqlglot.parse_one(sql_query, read=dialect)
        
        # Extract the table names from the parsed AST
        table_names = [table.name for table in parsed.find_all(sqlglot.expressions.Table)]
        
        return table_names
    except Exception as e:
        raise(e)


def extract_column_names(sql_query, dialect="sqlite"):
    """
    Extracts column names from an SQL query using sqlglot, with optional dialect specification.
    Handles PRAGMA, CREATE, INSERT, UPDATE, ALTER, SELECT (with WHERE clause), and other common SQL queries.

    Parameters:
        sql_query (str): The SQL query string.
        dialect (str): The SQL dialect to use for parsing (default is "sqlite").

    Returns:
        list: A list of column names found in the query.
    """
    try:
        column_names = []
        
        # Check for PRAGMA table_info query
        pragma_match = re.match(r"PRAGMA\s+table_info\((\w+)\)", sql_query.strip(), re.IGNORECASE)
        if pragma_match:
            # For PRAGMA table_info, return an empty list since no columns are actually mentioned in the query
            return []

        # Check for CREATE TABLE queries and extract column names
        create_match = re.match(r"CREATE\s+TABLE\s+\w+\s*\((.*)\)", sql_query.strip(), re.IGNORECASE)
        if create_match:
            column_definitions = create_match.group(1)
            column_names = [col.strip().split()[0] for col in column_definitions.split(",") if col.strip()]
            return column_names
        
        # Check for INSERT INTO queries and extract column names
        insert_match = re.match(r"INSERT\s+INTO\s+\w+\s*\((.*)\)", sql_query.strip(), re.IGNORECASE)
        if insert_match:
            insert_match_group = insert_match.group(1)
            #only take everything up to the first closing bracket, which highlights the column names
            insert_match_group = insert_match_group.split(")")[0]
            column_names = [col.strip() for col in insert_match_group.split(",")]
            return column_names
        
        # Check for UPDATE queries and extract column names
        update_match = re.match(r"UPDATE\s+\w+\s+SET\s+(.*)", sql_query.strip(), re.IGNORECASE)
        if update_match:
            set_clause = update_match.group(1)
            column_names = [item.split('=')[0].strip() for item in set_clause.split(",")]
            return column_names
        
        # Check for ALTER TABLE queries and extract column names
        alter_match = re.match(r"ALTER\s+TABLE\s+\w+\s+ADD\s+COLUMN\s+(\w+)", sql_query.strip(), re.IGNORECASE)
        if alter_match:
            return [alter_match.group(1)]
        
        # Check for SELECT queries and extract column names (including WHERE clause)
        select_match = re.match(r"SELECT\s+(.*?)\s+FROM", sql_query.strip(), re.IGNORECASE)
        if select_match:
            select_clause = select_match.group(1).strip()
            # If '*' is present, exclude it (we want actual column names)
            if select_clause != "*":
                # Split the columns by commas, ignoring functions or aliases
                columns = [col.split()[0] for col in select_clause.split(",") if col.strip() != "*"]
                return columns
        
        # Otherwise, parse the SQL query using sqlglot for general SQL queries
        parsed = sqlglot.parse_one(sql_query, read=dialect)
        
        # Extract column names from the parsed AST (if available)
        columns = parsed.find_all(sqlglot.expressions.Column)
        column_names = [column.name for column in columns]
        
        return column_names
    
    except Exception as e:
        raise(e)



def extract_column_value_mapping(sql_query, dialect="sqlite"):
    """
    Extracts values mentioned in an SQL query and maps them to corresponding column names.
    Handles PRAGMA, CREATE, INSERT, UPDATE, ALTER, SELECT, and other common SQL queries.

    Parameters:
        sql_query (str): The SQL query string.
        dialect (str): The SQL dialect to use for parsing (default is "sqlite").

    Returns:
        dict: A dictionary mapping column names to the corresponding values.
    """
    try:
        column_value_mapping = {}

        # Check for PRAGMA table_info query (no column values here)
        pragma_match = re.match(r"PRAGMA\s+table_info\((\w+)\)", sql_query.strip(), re.IGNORECASE)
        if pragma_match:
            return column_value_mapping  # No values to extract from PRAGMA

        # Check for CREATE TABLE queries (no values in the CREATE statement)
        create_match = re.match(r"CREATE\s+TABLE\s+\w+\s*\((.*)\)", sql_query.strip(), re.IGNORECASE)
        if create_match:
            return column_value_mapping  # CREATE TABLE does not have values

        # Check for INSERT INTO queries and extract values
        insert_match = re.match(r"INSERT\s+INTO\s+\w+\s*\((.*)\)\s+VALUES\s*\((.*)\)", sql_query.strip(), re.IGNORECASE)
        if insert_match:
            columns = [col.strip() for col in insert_match.group(1).split(",")]
            values = [val.strip().replace("'", "") for val in insert_match.group(2).split(",")]
            # Map columns to their corresponding values
            column_value_mapping = dict(zip(columns, values))
            return column_value_mapping

        # Check for UPDATE queries and extract values
        update_match = re.match(r"UPDATE\s+\w+\s+SET\s+(.*)", sql_query.strip(), re.IGNORECASE)
        if update_match:
            set_clause = update_match.group(1)
            # Extract column-value pairs from SET clause
            set_pairs = [item.split('=') for item in set_clause.split(",")]
            for pair in set_pairs:
                if len(pair) == 2:
                    column_value_mapping[pair[0].strip()] = pair[1].strip().replace("'", "")
        
        # Extract WHERE clause values for UPDATE, SELECT queries
        where_match = re.search(r"WHERE\s+(.*)", sql_query.strip(), re.IGNORECASE)
        if where_match:
            where_clause = where_match.group(1)
            # Split conditions in WHERE clause (supports AND conditions)
            conditions = re.split(r'\s+AND\s+|\s+OR\s+', where_clause, flags=re.IGNORECASE)
            for condition in conditions:
                if '=' in condition:  # Only handle simple column = value for now
                    col, val = condition.split('=', 1)
                    column_value_mapping[col.strip()] = val.strip().replace("'", "")
                elif '>' in condition or '<' in condition:
                    # Handle column > value or column < value
                    operator_match = re.search(r'(.*?)([><]=?|=)(.*)', condition)
                    if operator_match:
                        col = operator_match.group(1).strip()
                        val = operator_match.group(3).strip()
                        column_value_mapping[col] = val.replace("'", "")

        # Check for ALTER TABLE queries (no values in definitions)
        if "ALTER TABLE" in sql_query.upper():
            return column_value_mapping  # No data values expected here

        # Parse using sqlglot for general SQL handling
        parsed = sqlglot.parse_one(sql_query, read=dialect)

        # Extract column names and corresponding values from the parsed AST
        columns = parsed.find_all(sqlglot.expressions.Column)
        literals = parsed.find_all(sqlglot.expressions.Literal)

        # Map each column to its corresponding literal value
        for column, literal in zip(columns, literals):
            if hasattr(literal, 'value'):
                column_value_mapping[column.name] = literal.value.replace("'", "")
            else:
                column_value_mapping[column.name] = str(literal).replace("'", "")  # Fallback to string representation

        return column_value_mapping
    
    except Exception as e:
        raise(e)
        #return f"Error parsing SQL query: {e}"
    


def analyse_sql_query(sql_query, dialect="sqlite"):
    """
    Analyses an SQL query and extracts table names, column names, and column-value mappings.

    Parameters:
        sql_query (str): The SQL query string.
        dialect (str): The SQL dialect to use for parsing (default is "sqlite").

    Returns:
        dict: A dictionary containing table names, column names, and column-value mappings.
    """
    # Check if the input is valid
    if not isinstance(sql_query, str) or not sql_query.strip():
        raise ValueError("The SQL query must be a non-empty string.")
    
    try:
        # Extract table names
        table_names = extract_table_names(sql_query, dialect=dialect)
        
        # Extract column names
        column_names = extract_column_names(sql_query, dialect=dialect)
        
        # Extract column-value mapping
        column_value_mapping = extract_column_value_mapping(sql_query, dialect=dialect)

        # Combine results into a dictionary
        analysis_result = {
            "table_names": table_names,
            "column_names": column_names,
            "column_value_mapping": column_value_mapping
        }
        
        return analysis_result

    except ValueError as ve:
        # Likely caused by invalid or unexpected query input
        raise ValueError(f"ValueError while analyzing SQL query: {ve}")
    
    except sqlglot.errors.ParseError as pe:
        # Handle SQL parsing errors specifically
        raise ValueError(f"ParseError: Unable to parse the SQL query. Details: {pe}")
    
    except Exception as e:
        # Catch-all for unexpected errors (should rarely happen)
        raise RuntimeError(f"Unexpected error occurred: {e}")
    
