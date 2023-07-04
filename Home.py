import streamlit as st
import function as fn


# model = pickle.load(open('model_rangga.pkl', 'rb'))
# model = joblib.load



with st.form(key='my_form', clear_on_submit=True):
    files = st.file_uploader(label='Insert Image', type=['jpg'], accept_multiple_files=True)
    submit_button = st.form_submit_button(label='Submit')

    if not files and submit_button:
        st.warning('Please insert image')

    elif files and submit_button:
        st.success('Success')
        for file in files:
            # name = fn.save_files(file)
            image, height, width = fn.load_image(file)

            prediction = fn.predict(file=image)

            st.image(fn.cv2.resize(image,(width,height)))
            st.write(f'{prediction}')

    # fn.del_file()

# st.markdown('<h1>halo</h1>', unsafe_allow_html=True)

# css = """
# <style>
# [data-testid="stMarkdownContainer"]{
#     background-color: #ffffff;
# }


# </style>
# """

# st.markdown(css, unsafe_allow_html=True)