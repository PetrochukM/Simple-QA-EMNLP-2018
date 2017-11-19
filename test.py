import psycopg2
import psycopg2.extras

# Load .env file
pass_ = {}
for line in open('.pass'):
    split = line.strip().split('=')
    pass_[split[0]] = split[1]

# Connect
connection = psycopg2.connect(
    dbname=pass_['DB_NAME'],
    port=pass_['DB_PORT'],
    user=pass_['DB_USER'],
    host=pass_['DB_HOST'],
    password=pass_['DB_PASS'])
cursor = connection.cursor()

print('here')
cursor.execute("""
    CREATE TABLE fb_kg (
        object_mid varchar NOT NULL,
        relation varchar NOT NULL,
        subject_mid varchar NOT NULL,
        PRIMARY KEY(object_mid, relation, subject_mid)
    );""")
print('here')