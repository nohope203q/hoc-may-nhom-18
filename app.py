import streamlit as st
import pickle
import numpy as np
from PIL import Image
import pandas as pd

try:
    from ensemble_models import (
        RandomForest,
        VotingClassifier,
        XGBoostCustom,
        AdaBoostMulticlass  
    )
    CUSTOM_MODELS_AVAILABLE = True
except ImportError:
    CUSTOM_MODELS_AVAILABLE = False
    st.warning("Không tìm thấy ensemble_models.py. Vui lòng upload file này!")
# Cấu hình trang
st.set_page_config(
    page_title="Phân loại Hoa Iris",
    page_icon="🌸",
    layout="wide"
)

# Tiêu đề
st.title("🌸 Ứng dụng Phân loại Hoa Iris")
st.markdown("### Sử dụng Ensemble Learning")
st.markdown("---")

# Sidebar để chọn model
st.sidebar.header("⚙️ Cấu hình")
model_choice = st.sidebar.selectbox(
    "Chọn Model:",
    [
        "Random Forest (Bagging)", 
        "Voting Classifier", 
        "XGBoost Custom (Boosting)",
        "AdaBoost (Boosting)"  
    ]
)

# Load model
@st.cache_resource
def load_model(model_name):
    try:
        if model_name == "Random Forest (Bagging)":
            with open('model_rf_custom.pkl', 'rb') as f:
                model = pickle.load(f)
        elif model_name == "Voting Classifier":
            with open('model_voting_custom.pkl', 'rb') as f:
                model = pickle.load(f)
        elif model_name == "XGBoost Custom (Boosting)":
            with open('model_xgboost_custom.pkl', 'rb') as f:
                model = pickle.load(f)
        else:  # AdaBoost (Boosting)
            with open('model_adaboost_custom.pkl', 'rb') as f:
                model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"Không tìm thấy file model! Vui lòng chạy train_models.py trước")
        return None
    except Exception as e:
        st.error(f"Lỗi khi load model: {str(e)}")
        return None
iris_names = {
    0: "Iris Setosa",
    1: "Iris Versicolor",
    2: "Iris Virginica"
}
iris_descriptions = {
    0: "**Iris Setosa**: Loài hoa nhỏ nhất, cánh hoa màu tím nhạt đến xanh dương, dễ phân biệt nhất.",
    1: "**Iris Versicolor**: Loài hoa cỡ trung bình, cánh hoa màu tím, có vân màu trắng và vàng.",
    2: "**Iris Virginica**: Loài hoa lớn nhất, cánh hoa màu tím đậm đến xanh nhạt, thường cao nhất."
}

# Load hình ảnh
@st.cache_data
def load_images():
    images = {}
    try:
        images[0] = Image.open('iris_setosa.jpg')
        images[1] = Image.open('iris_versicolor.jpg')
        images[2] = Image.open('iris_virginica.jpg')
    except:
        st.warning("Không tìm thấy một số hình ảnh. Vui lòng upload hình ảnh hoa Iris.")
    return images

# Giao diện nhập liệu
st.header("📊 Nhập thông số hoa Iris")
col1, col2 = st.columns(2)
with col1:
    sepal_length = st.number_input(
        "Chiều dài đài hoa (cm)",
        min_value=0.0,
        max_value=10.0,
        value=5.1,
        step=0.1,
        help="Thông thường từ 4.3 đến 7.9 cm"
    )
    sepal_width = st.number_input(
        "Chiều rộng đài hoa (cm)",
        min_value=0.0,
        max_value=10.0,
        value=3.5,
        step=0.1,
        help="Thông thường từ 2.0 đến 4.4 cm"
    )
with col2:
    petal_length = st.number_input(
        "Chiều dài cánh hoa (cm)",
        min_value=0.0,
        max_value=10.0,
        value=1.4,
        step=0.1,
        help="Thông thường từ 1.0 đến 6.9 cm"
    )
    petal_width = st.number_input(
        "Chiều rộng cánh hoa (cm)",
        min_value=0.0,
        max_value=10.0,
        value=0.2,
        step=0.1,
        help="Thông thường từ 0.1 đến 2.5 cm"
    )

# Nút dự đoán
if st.button("🔍 Dự đoán loài hoa", type="primary"):
    if not CUSTOM_MODELS_AVAILABLE:
        st.error("Cần file ensemble_models.py để chạy!")
        st.stop()
    model = load_model(model_choice)

    if model is not None:
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        try:
            prediction = model.predict(input_data)[0]
            prediction = int(prediction)
            if prediction < 0:
                prediction = 0
            elif prediction > 2:
                prediction = 2
            st.markdown("---")
            st.success("✅ Dự đoán hoàn tất!")
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("🎯 Kết quả dự đoán")
                st.markdown(f"### {iris_names[prediction]}")
                st.info(iris_descriptions[prediction])
                st.markdown("#### 📝 Thông số đã nhập:")
                data_df = pd.DataFrame({
                    'Thông số': ['Dài đài', 'Rộng đài', 'Dài cánh', 'Rộng cánh'],
                    'Giá trị (cm)': [sepal_length, sepal_width, petal_length, petal_width]
                })
                st.table(data_df)
                if hasattr(model, 'predict_proba'):
                    try:
                        probabilities = model.predict_proba(input_data)[0]
                        if len(probabilities) < 3:
                            probabilities = np.append(probabilities, [0] * (3 - len(probabilities)))
                        st.markdown("#### 📊 Xác suất dự đoán:")
                        prob_df = pd.DataFrame({
                            'Loài hoa': [iris_names[i] for i in range(3)],
                            'Xác suất': [f"{prob*100:.2f}%" for prob in probabilities[:3]]
                        })
                        st.table(prob_df)
                        st.bar_chart(
                            pd.DataFrame(
                                probabilities[:3],
                                index=[iris_names[i] for i in range(3)],
                                columns=['Xác suất']
                            )
                        )
                    except:
                        pass
            with col2:
                st.subheader("🌸 Hình ảnh loài hoa")
                images = load_images()
                if prediction in images:
                    st.image(images[prediction], use_container_width=True)
                else:
                    st.warning("Chưa có hình ảnh cho loài hoa này")
        except Exception as e:
            st.error(f"Lỗi khi dự đoán: {str(e)}")
            st.info("Vui lòng kiểm tra lại model và thử lại.")

# Thông tin về dataset
with st.expander("ℹ️ Thông tin về Dataset Iris"):
    st.markdown("""
    **Dataset Iris** là một trong những dataset kinh điển nhất trong Machine Learning, được giới thiệu bởi Ronald Fisher năm 1936.

    **Đặc điểm:**
    - 150 mẫu hoa Iris
    - 3 loài: Setosa, Versicolor, Virginica (mỗi loài 50 mẫu)
    - 4 đặc trưng: Chiều dài/rộng đài hoa và cánh hoa

    **Phạm vi giá trị thông thường:**
    - Chiều dài đài: 4.3 - 7.9 cm
    - Chiều rộng đài: 2.0 - 4.4 cm
    - Chiều dài cánh: 1.0 - 6.9 cm
    - Chiều rộng cánh: 0.1 - 2.5 cm
    """)

# Thông tin về models
with st.expander("Về các Models"):
    st.markdown("""
    **3 Models được sử dụng:**
    
    **1. Random Forest (Bagging)**
    - 20 decision trees được train độc lập
    - Bootstrap sampling với hoàn lại
    - Majority voting để dự đoán
    - Giảm variance, tránh overfitting
    
    **2. Voting Classifier**
    - Kết hợp 3 models: Decision Tree, Logistic Regression, KNN
    - Hard voting (majority vote)
    - Tận dụng diverse models
    
    **3. XGBoost Custom (Gradient Boosting)**
    - 50 sequential trees
    - Fit residuals của model trước
    - Learning rate = 0.1
    - Giảm bias, cải thiện accuracy
    
    **4. AdaBoost (Boosting)**
    - 50 weak learners (decision stumps)
    - One-vs-Rest strategy cho multiclass
    - Adaptive weighting
    - Sequential training với error-based reweighting
                
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🎓 Đồ án cuối kỳ môn Machine Learning</p>
        <p>📚 Đề tài: Phân loại Hoa Iris với Ensemble Models</p>
    </div>
    """,
    unsafe_allow_html=True
)