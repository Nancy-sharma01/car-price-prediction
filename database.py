import mysql.connector
import streamlit as st


def get_connection():

    st.write("DEBUG HOST:", st.secrets["mysql"]["host"])
    st.write("DEBUG PORT:", st.secrets["mysql"]["port"])

    return mysql.connector.connect(
        host=st.secrets["mysql"]["host"],
        port=st.secrets["mysql"]["port"],
        user=st.secrets["mysql"]["user"],
        password=st.secrets["mysql"]["password"],
        database=st.secrets["mysql"]["database"],
        ssl_disabled=False
    )