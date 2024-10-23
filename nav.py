


import streamlit as st

import streamlit.components.v1 as components


nav =st.navigation({"": [st.Page('home.py', title='Home', icon=':material/home:'),
                    st.Page('about.py', title='About', icon=':material/face:'),
                    st.Page('education.py', title='Education', icon=':material/school:')
                    ],
                    "Projects": [st.Page('land_value.py', title='Land value - Analysis', icon=':material/developer_board:'),
                    st.Page('lv_predict.py', title='Land value - Prediction', icon=':material/developer_board:'),
                    st.Page('predict_disease.py', title='Disease prediction', icon=':material/developer_board:'),
                    st.Page('tips.py', title='Tips - Analysis', icon=':material/developer_board:'),
                    st.Page('tips_predict.py', title='Tips - Prediction', icon=':material/developer_board:'),
                    st.Page('Blood_Donation_Analysis.py', title='Blood Donation - Analysis', icon=':material/developer_board:')
                    ],
                    })

with st.sidebar:
        components.iframe('https://lottie.host/embed/11baf9d0-2f09-4940-8fc4-8545070fae47/PlZnidWzAu.json')
        #navbar
nav.run()