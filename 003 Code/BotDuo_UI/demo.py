import streamlit as st
from PIL import Image, ImageFilter
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from arch.iformer.inception_transformer import iformer_small

# Wide layout 설정
st.set_page_config(
    page_title="Steganalysis Framework",
    # layout="wide",  # 페이지를 wide 레이아웃으로 설정
    initial_sidebar_state="expanded"  # 사이드바 초기 상태
)

# 이미지 전처리 함수
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)  # 배치 차원 추가

def inference(image_path, model, device): 
    model.eval()
    image = preprocess_image(image_path).to(device)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
        predicted_class = np.argmax(probs)
    return probs, predicted_class

# 모델 추론 함수
def model_inference(image_path, model_path, device=torch.device("cpu")):
    # 이미지 전처리 및 추론
    # image_tensor = preprocess_image(image)
    # image_tensor = image_tensor.to(device)
    model = iformer_small(pretrained=True, num_classes=2).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
      
    image = preprocess_image(image_path).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
        predicted_class = np.argmax(probs)

    return {'Cover': probs[0], 'Stego': probs[1]}

def compute_residual(image1, image2, operation="subtract"):
    """
    두 이미지의 잔차를 계산합니다.
    operation: "subtract" 또는 "invert".
        - "subtract": image1 - image2
        - "invert": image2 - image1
    """
    image1_array = np.array(image1)
    image2_array = np.array(image2)
    
    if operation == "subtract":
        residual_array = np.abs(image1_array - image2_array)
    elif operation == "invert":
        residual_array = np.abs(image2_array - image1_array)
    else:
        raise ValueError("Invalid operation type. Use 'subtract' or 'invert'.")
    
    return Image.fromarray(residual_array)



def model_inference_with_gradcam(image_path, model_path, device=torch.device("cpu")):
    """
    GradCAM을 사용한 모델 추론
    """
    model = iformer_small(pretrained=True, num_classes=2).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # 이미지 전처리
    input_tensor = preprocess_image(image_path).to(device)

    # 추론
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
        predicted_class = np.argmax(probs)
        
def go_to_home():
    # 상태 초기화
    st.session_state.page = 'home'
    st.session_state.selected_category = None
    st.session_state.selected_device = None
    st.session_state.selected_image_name = None
    st.session_state.model = None
    st.session_state.inference_done = False
    
def select_image_and_reset_inference_done(img_name, image_type):
    st.session_state.selected_image_name = img_name
    st.session_state.image_type = image_type  # 선택한 이미지 타입 저장
    st.session_state.page = 'selected_image'
    st.session_state.inference_done = False

def main():
    st.markdown(
    """
    <h2 style="text-align: center; font-size: 40px;">스테그어날리시스를 위한 모델 개발</h2>
    """,
    unsafe_allow_html=True
)

    # 초기 상태 설정
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    if 'selected_category' not in st.session_state:
        st.session_state.selected_category = None
    if 'selected_device' not in st.session_state:
        st.session_state.selected_device = None
    if 'selected_image_name' not in st.session_state:
        st.session_state.selected_image_name = None
    if 'model' not in st.session_state:
        st.session_state.model = None

    # 페이지 조건별 렌더링
    if st.session_state.page == 'home':
        
        # 도메인 선택 및 다음 버튼
        category = st.selectbox('도메인 선택:', ['압축영역', '공간영역'])

        if category:
            st.session_state.selected_category = category
            model_dir = 'models/'  # 모델 디렉터리 설정
            category_dir = os.path.join(model_dir, category)
            model_files = [f for f in os.listdir(category_dir) if f.endswith('.pth') or f.endswith('.pth.tar')]

            if model_files:
                # 첫 번째 모델 파일을 기본으로 선택
                model_path = os.path.join(category_dir, model_files[0])
                st.session_state.model = model_path  # 모델 경로 저장
            else:
                st.error(f"{category} 폴더에 모델 파일이 없습니다.")
                return

            st.button('다음', on_click=lambda: st.session_state.update(page='device_select'))

             # 페이지 중앙에 이미지 표시
            framework_image_path = os.path.join('images', 'framework.png')
            if os.path.exists(framework_image_path):
                # 이미지 중앙 정렬
                with st.container():
                    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                    st.image(framework_image_path, caption="Framework Overview", use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.error("framework.png 파일이 없습니다. 이미지를 확인하세요.")

    elif st.session_state.page == 'device_select':
        selected_category = st.session_state.selected_category
        image_dir = 'images/'
        category_dir = os.path.join(image_dir, selected_category)

        # 카테고리에 따라 기기 이름에 '*' 추가
        devices = [d for d in os.listdir(category_dir) if os.path.isdir(os.path.join(category_dir, d))]
        if selected_category == '공간영역':
            devices = [d + ' *' if d in ['Galaxy_Note9', 'Galaxy_S21'] else d for d in devices]
        elif selected_category == '압축영역':
            devices = [d + ' *' if d in ['Galaxy_S22', 'iPhone_13mini'] else d for d in devices]

        # 기기 선택 박스
        selected_device = st.selectbox('모바일 기기 선택:', devices)

        # 선택 바 아래에 설명 추가
        st.markdown(
            "<p style='font-size: 14px; color: grey;'>* 가 붙은 기기는 학습에 사용하지 않은 기기입니다.</p>",
            unsafe_allow_html=True
        )

        if selected_device:
            # '*'를 제거하고 저장 (필요 시)
            st.session_state.selected_device = selected_device.replace(' *', '')
            st.button('다음', on_click=lambda: st.session_state.update(page='image_select'))
        st.button('이전', on_click=go_to_home)

    elif st.session_state.page == 'image_select':
        selected_category = st.session_state.selected_category
        selected_device = st.session_state.selected_device
        stego_dir = os.path.join('images/', selected_category, selected_device, 'Stego')
        cover_dir = os.path.join('images/', selected_category, selected_device, 'Cover')

        # Cover 이미지 섹션
        st.markdown("### Cover 이미지")
        cover_images = [img for img in os.listdir(cover_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        if cover_images:
            cover_cols = st.columns(len(cover_images))
            for idx, img_name in enumerate(cover_images):
                with cover_cols[idx]:
                    image_path = os.path.join(cover_dir, img_name)
                    image = Image.open(image_path).resize((224, 224))
                    st.image(image, width=224)
                    st.button(
                        "선택",
                        key=f"cover_{img_name}",
                        on_click=select_image_and_reset_inference_done,
                        args=(img_name, "cover")
                    )
        else:
            st.write("Cover 이미지가 없습니다.")

        st.markdown("---")

        # Stego 이미지 섹션
        st.markdown("### Stego 이미지")
        stego_images = [img for img in os.listdir(stego_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        if stego_images:
            stego_cols = st.columns(len(stego_images))
            for idx, img_name in enumerate(stego_images):
                with stego_cols[idx]:
                    image_path = os.path.join(stego_dir, img_name)
                    image = Image.open(image_path).resize((224, 224))
                    st.image(image, width=224)
                    st.button(
                        "선택",
                        key=f"stego_{img_name}",
                        on_click=select_image_and_reset_inference_done,
                        args=(img_name, "stego")
                    )
        else:
            st.write("Stego 이미지가 없습니다.")

        st.button('이전', on_click=lambda: st.session_state.update(page='device_select'))
        
    elif st.session_state.page == 'selected_image':
        selected_category = st.session_state.selected_category
        selected_device = st.session_state.selected_device
        selected_image_name = st.session_state.selected_image_name
        image_type = st.session_state.image_type  # 선택한 이미지 타입 (cover 또는 stego)

        # 이미지 경로 설정
        if image_type == "cover":
            image_dir = os.path.join('images/', selected_category, selected_device, 'Cover')
            residual_operation = "subtract"  # Cover 간 비교 시 subtract 사용
        else:
            image_dir = os.path.join('images/', selected_category, selected_device, 'Stego')
            other_image_dir = os.path.join('images/', selected_category, selected_device, 'Cover')

        selected_image_path = os.path.join(image_dir, selected_image_name)

        # 선택된 이미지 표시
        if not os.path.exists(selected_image_path):
            st.error("선택한 이미지가 없습니다.")
            return

        selected_image = Image.open(selected_image_path)

        # 잔차 계산
        if image_type == "cover":
            # Cover 이미지: 다른 Cover 이미지와 비교
            cover_images = [
                img for img in os.listdir(image_dir) if img != selected_image_name and img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
            ]
            if cover_images:
                # 첫 번째 다른 Cover 이미지 선택
                other_image_path = os.path.join(image_dir, cover_images[0])
                other_image = Image.open(other_image_path)
                residual_image = compute_residual(selected_image, selected_image, operation=residual_operation)
            else:
                st.error("대응하는 다른 Cover 이미지가 없습니다.")
                return
        else:
            # Stego 이미지: 대응하는 Cover 이미지와 비교
            other_image_path = os.path.join(other_image_dir, selected_image_name)
            if not os.path.exists(other_image_path):
                st.error("대응하는 Cover 이미지가 없습니다.")
                return
            other_image = Image.open(other_image_path)
            residual_image = compute_residual(selected_image, other_image)

        # 추론 결과 로드
        if 'inference_done' not in st.session_state:
            st.session_state.inference_done = False

        if not st.session_state.inference_done:
            # 추론 시작 버튼
            if st.button('추론 시작'):
                model = st.session_state.model
                if model is None:
                    st.error("모델이 로드되지 않았습니다.")
                    return

                # 모델 추론
                results = model_inference(selected_image_path, model)
                st.session_state.results = results  # 추론 결과를 상태에 저장
                st.session_state.inference_done = True

        # 추론 완료 상태에서 잔차 이미지와 확률 값 바로 표시
        if st.session_state.inference_done:
            results = st.session_state.results

            # 레이아웃: 추론 이미지, 잔차 이미지, 확률 값
            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                st.image(selected_image, caption=f"선택된 {image_type.capitalize()} 이미지", use_container_width=True)

            with col2:
                st.image(residual_image, caption="Residual Image", use_container_width=True)

            with col3:
                model_name = os.path.basename(st.session_state.model)  # 모델 파일 이름 추출
                algorithm = "nsf5" if st.session_state.selected_category == "압축영역" else "lsb"
                method = "CE 0.7 + SupCon 0.3" if st.session_state.selected_category == "압축영역" else "SAM 0.05"

                st.markdown(
                    f"""
                    **모델 정보**:
                    - Backbone: iFormer-S  
                    - Method: {method}  
                    - Data: 통합 데이터 20만장  
                    - Algorithm: {algorithm}
                    """
                )
                st.write(f"**커버 이미지 확률:** {results['Cover'] * 100:.2f}%")
                st.write(f"**스테고 이미지 확률:** {results['Stego'] * 100:.2f}%")



        # 이전 및 홈으로 버튼
        st.button('이전', on_click=lambda: st.session_state.update(page='image_select'))
        st.button('홈으로', on_click=go_to_home)




if __name__ == '__main__':
    main()