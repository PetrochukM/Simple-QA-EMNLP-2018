import psycopg2
import psycopg2.extras
import os


def get_connection():
    # Load .env file
    pass_ = {}

    # Get the path relative to the directory this file is in
    _directory_path = os.path.dirname(os.path.realpath(__file__))
    pass_path = os.path.join(_directory_path, '../.pass')
    for line in open(pass_path):
        split = line.strip().split('=')
        pass_[split[0]] = split[1]

    # Connect
    return psycopg2.connect(
        dbname=pass_['DB_NAME'],
        port=pass_['DB_PORT'],
        user=pass_['DB_USER'],
        host=pass_['DB_HOST'],
        password=pass_['DB_PASS'])