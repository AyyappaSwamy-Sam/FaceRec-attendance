import sqlite3
import pandas as pd
from datetime import date

# Connect to the database
conn = sqlite3.connect('face_attendance.db')

# Function to execute queries and print results
def run_query(query, description):
    print(f"\n--- {description} ---")
    try:
        result = pd.read_sql_query(query, conn)
        if len(result) > 0:
            print(result)
        else:
            print("No records found.")
    except Exception as e:
        print(f"Error: {e}")

# Show all tables
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Database Tables:", [table[0] for table in tables])

# Show users
run_query("SELECT * FROM users", "All Registered Users")

# Show today's attendance
today = date.today().strftime("%Y-%m-%d")
run_query(f"SELECT * FROM attendance WHERE date = '{today}'", f"Today's Attendance ({today})")

# Show counts
run_query("SELECT COUNT(*) AS total_users FROM users", "Total Registered Users")
run_query("SELECT COUNT(*) AS total_attendance_records FROM attendance", "Total Attendance Records")

# Show users with most attendance records
run_query("""
SELECT users.name, users.user_id, COUNT(attendance.id) AS attendance_count 
FROM users 
LEFT JOIN attendance ON users.user_id = attendance.user_id 
GROUP BY users.user_id 
ORDER BY attendance_count DESC
LIMIT 10
""", "Top 10 Users by Attendance")

# Close the connection
conn.close()
print("\nDatabase connection closed.")