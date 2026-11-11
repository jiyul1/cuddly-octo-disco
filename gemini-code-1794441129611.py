import streamlit as st
import pickle
import numpy as np
import Orange

# 1. 모델 파일 로드 (캐싱을 통해 매번 다시 로드하지 않도록 설정)
@st.cache_resource
def load_model():
    with open('traffic_casualty_model.pkcls', 'rb') as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("모델 파일('traffic_casualty_model.pkcls')을 찾을 수 없습니다. app.py와 같은 경로에 파일을 업로드해주세요.")
    st.stop()

st.title("🚦 실시간 교통사고 예상 사망자 예측 시스템")
st.write("지역과 사고 규모를 조절하여 예상 사망자 수를 확인하고 안전 수칙을 준수해주세요.")

# 2. 사이드바 UI 설정 (시도 선택 및 슬라이더)
st.sidebar.header("사고 데이터 입력")

# 예시: 학습된 데이터의 시도 리스트 (실제 모델 학습 시 사용된 인코딩 순서에 맞춰야 합니다)
sido_list = ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종", "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"]
sido = st.sidebar.selectbox("시도 선택", sido_list)

# 슬라이더 값 실시간 연동
accidents = st.sidebar.slider("사고건수", min_value=0, max_value=1000, value=100, step=10)
minor_injuries = st.sidebar.slider("경상자수", min_value=0, max_value=2000, value=150, step=10)

# 시도 범주형 데이터 숫자로 변환 (모델 학습 시 사용한 인코딩 규칙에 맞게 수정 필요)
sido_encoded = sido_list.index(sido) 

# Orange3 모델 예측을 위한 2D Numpy 배열 변환
# 주의: [시도, 사고건수, 경상자수] 순서는 실제 모델 학습 시 입력된 피처(Feature) 순서와 완벽히 동일해야 합니다.
input_data = np.array([[sido_encoded, accidents, minor_injuries]])

# 모델 예측 함수 실행
try:
    # Orange3 모델은 Callable 객체로 예측 수행
    prediction = model(input_data)[0]
    # 결과값이 음수가 나오지 않도록 보정
    prediction = max(0, prediction)
except Exception as e:
    prediction = 0
    st.sidebar.error(f"예측 오류: 데이터 형식을 확인하세요.\n{e}")

# 3. 계산된 숫자에 따른 상태 판별 및 대형 신호등 출력
st.markdown("---")
st.subheader(f"📊 실시간 예상 사망자 수: **{prediction:.2f} 명**")

if prediction < 1.0:
    light_color = "#28a745" # 초록색
    light_icon = "🟢"
    light_text = "안전"
elif prediction < 3.0:
    light_color = "#ffc107" # 노란색
    light_icon = "🟡"
    light_text = "주의"
else:
    light_color = "#dc3545" # 빨간색
    light_icon = "🔴"
    light_text = "위험"

# HTML/CSS를 활용한 대형 신호등 카드 UI
st.markdown(
    f"""
    <div style="
        display: flex; 
        align-items: center; 
        justify-content: center; 
        background-color: {light_color}; 
        padding: 30px; 
        border-radius: 15px; 
        color: white; 
        font-size: 40px; 
        font-weight: bold; 
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 30px;">
        {light_icon} {light_text} 구간
    </div>
    """,
    unsafe_allow_html=True
)

# 4. 어린이 보호구역 안전 수칙 카드
st.markdown("### 🚸 어린이 보호구역 필수 안전 수칙")
st.info(
    """
    **1. 규정 속도 절대 준수**  
    스쿨존 내에서는 시속 30km 이하로 서행하세요. 속도를 줄이는 것만으로도 대형 사고를 막을 수 있습니다.
    
    **2. 횡단보도 앞 일단 멈춤**  
    보행자가 없더라도 횡단보도 앞에서는 무조건 일시 정지 후 주위를 살피세요.
    
    **3. 불법 주정차 금지**  
    학교 주변 불법 주차 차량은 키가 작은 어린이들의 시야를 가려 치명적인 사고를 유발합니다.
    
    **4. 급제동 및 급출발 자제**  
    어린이는 돌발 행동이 잦습니다. 언제든 멈출 수 있도록 방어 운전을 생활화하세요.
    """
)