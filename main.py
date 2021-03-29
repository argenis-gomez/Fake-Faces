import streamlit as st
from utils import load_model, return_prediction
from PIL import Image


def main():
    st.title('Generador de rostros artificiales')

    st.write("A continuación podrás generar rostros artificiales.")

    generar = st.button('Generar')

    if generar:
        st.write('Generando rostro...')

        image_generated = return_prediction(MODEL)
        image_generated = Image.fromarray(image_generated.astype('uint8'))

        st.image(image_generated, width=300)


if __name__ == '__main__':
    MODEL = load_model('ckpt/generator/model')
    main()
