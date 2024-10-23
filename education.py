import streamlit as st

def education_page():
    st.title("Education")
    with st.container(border=True):
        st.write(
            """
            **EFREI Paris** | 2021-2026
            - 5-year computer engineering school. Currently in the second year of the engineering cycle (M1).
            - Courses taken: Advanced databases, Data visualization, Machine learning, Big Data Frameworks.
            """
        )
    with st.container(border=True):
        st.write(
            '''
            **APU Kuala Lumpur** | 2023
            - Study semester in Malaysia
            - Courses taken: Advanced web programming, Java programming, Object-oriented Analysis and Design with UML
            '''
        )
    with st.container(border=True):
        st.write(
            """
            **Institution des Chartreux** | 2006-2021
            - High school diploma with a focus on math, physics and computer science with a grade of Bien
            """
        )
    

education_page()