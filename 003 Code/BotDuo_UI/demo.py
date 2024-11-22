import streamlit as st
from PIL import Image, ImageFilter
import os
import numpy as np

def model_inference(image):
    cover_prob = np.random.rand()
    stego_prob = 1 - cover_prob
    return {'Cover': cover_prob, 'Stego': stego_prob}

def compute_residual(image):
    blurred_image = image.filter(ImageFilter.GaussianBlur(radius=2))
    residual = Image.fromarray(np.abs(np.array(image).astype('int32') - np.array(blurred_image).astype('int32')).astype('uint8'))
    return residual

def main():
    st.title("스테그어날리시스를 위한 모델 개발")

    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    if 'selected_device' not in st.session_state:
        st.session_state.selected_device = None
    if 'selected_image_name' not in st.session_state:
        st.session_state.selected_image_name = None

    def go_to_image_select():
        st.session_state.page = 'image_select'

    def go_to_selected_image():
        st.session_state.page = 'selected_image'
        
    def go_to_inference():
        st.session_state.page = 'inference'

    def go_to_home():
        st.session_state.page = 'home'
        st.session_state.selected_device = None
        st.session_state.selected_image_name = None

    def go_to_previous_page():
        if st.session_state.page == 'select_image':
            st.session_state.page = 'home'
            st.session_state.selected_device = None
        elif st.session_state.page == 'inference':
            st.session_state.page = 'select_image'
            st.session_state.selected_image_name = None

    image_dir = 'images/'

    if st.session_state.page == 'home':
        st.write("학습된 모델의 소스 데이터를 선택하세요.")

        devices = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]

        selected_device = st.selectbox('모바일 기기 선택:', devices)

        if selected_device:
            st.session_state.selected_device = selected_device
            st.button('다음', on_click=go_to_image_select)

    elif st.session_state.page == 'image_select':
        st.write("이미지를 선택하세요.")

        selected_device = st.session_state.selected_device
        device_image_dir = os.path.join(image_dir, selected_device)
        images = [img for img in os.listdir(device_image_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

        if images:
            num_cols = 3 
            cols = st.columns(num_cols)

            for idx, img_name in enumerate(images):
                image_path = os.path.join(device_image_dir, img_name)
                image = Image.open(image_path)

                thumbnail_image = image.resize((224, 224))

                col = cols[idx % num_cols]
                with col:
                    st.image(thumbnail_image, caption=img_name, width=224)
                    def select_image(img=img_name):
                        st.session_state.selected_image_name = img
                        st.session_state.page = 'inference'
                    st.button('선택', key=img_name, on_click=select_image)

            st.button('이전', on_click=go_to_previous_page)

        else:
            st.write("선택한 기기에 이미지가 없습니다.")
            st.button('이전', on_click=go_to_previous_page)

    elif st.session_state.page == 'selected_image':
        selected_device = st.session_state.selected_device
        selected_image_name = st.session_state.selected_image_name
        device_image_dir = os.path.join(image_dir, selected_device)
        image_path = os.path.join(device_image_dir, selected_image_name)
        image = Image.open(image_path)

        st.write("선택된 이미지:")
        st.image(image, caption='선택된 이미지', use_container_width=True)

        if 'inference_done' not in st.session_state:
            st.session_state.inference_done = False

        if not st.session_state.inference_done:
            if st.button('추론 시작'):
                results = model_inference(image)
                st.write("3. 추론 결과를 출력합니다.")
                st.write(f"**커버 이미지일 확률:** {results['Cover']:.4f}")
                st.write(f"**스테고 이미지일 확률:** {results['Stego']:.4f}")
                
                residual_image = compute_residual(image)
                st.image(residual_image, caption='삽입된 메세지 이미지', use_container_width=True)

                st.button('홈으로', on_click=go_to_home)
                st.session_state.inference_done = True
            else:
                st.button('이전', on_click=go_to_previous_page)

    else:
        st.write("잘못된 페이지 상태입니다.")
        st.button('홈으로', on_click=go_to_home)

if __name__ == '__main__':
    main()
