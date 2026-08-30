import bcrypt
from database import get_connection


def signup(username, email, password):

    conn = get_connection()
    cursor = conn.cursor()

    password_hash = bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt()
    )

    query = """
        INSERT INTO users (username, email, password_hash)
        VALUES (%s, %s, %s)
    """

    cursor.execute(
        query,
        (username, email, password_hash.decode("utf-8"))
    )

    conn.commit()

    cursor.close()
    conn.close()

def login(email, password):

    conn = get_connection()
    cursor = conn.cursor()

    query = """
        SELECT password_hash
        FROM users
        WHERE email = %s
    """

    cursor.execute(query, (email,))

    result = cursor.fetchone()

    cursor.close()
    conn.close()

    if result is None:
        return False

    stored_hash = result[0]

    return bcrypt.checkpw(
        password.encode("utf-8"),
        stored_hash.encode("utf-8")
    )