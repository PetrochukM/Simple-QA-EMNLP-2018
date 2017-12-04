import psycopg2
import psycopg2.extras


def get_connection():
    # Load .env file
    pass_ = {}
    for line in open('../../.pass'):
        split = line.strip().split('=')
        pass_[split[0]] = split[1]

    # Connect
    return psycopg2.connect(
        dbname=pass_['DB_NAME'],
        port=pass_['DB_PORT'],
        user=pass_['DB_USER'],
        host=pass_['DB_HOST'],
        password=pass_['DB_PASS'])