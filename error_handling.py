import streamlit as st
import traceback

def handle_error(error):
    st.error(f"An error occurred: {str(error)}")
    st.write("Error details:")
    st.code(traceback.format_exc())

