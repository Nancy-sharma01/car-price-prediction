import mysql.connector
import streamlit as st


def get_connection():

    host = st.secrets["mysql"]["host"]
    port = st.secrets["mysql"]["port"]

    st.write("DEBUG HOST:", host)
    st.write("DEBUG PORT:", port)

    return mysql.connector.connect(
        host=host,
        port=port,
        user=st.secrets["mysql"]["user"],
        password=st.secrets["mysql"]["password"],
        database=st.secrets["mysql"]["database"],
        ssl_disabled=False,
        connection_timeout=10
    )