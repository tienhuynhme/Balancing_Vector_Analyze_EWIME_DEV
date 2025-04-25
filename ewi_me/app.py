import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("⚙️ Two-Plane Dynamic Balancing Tool (EWI Edition)")

st.markdown("### 🧾 Input Parameters")

col1, col2 = st.columns(2)

with col1:
    dA = st.number_input("📏 Khoảng cách từ gối đến mặt phẳng A (mm)", value=100.0)
    dB = st.number_input("📏 Khoảng cách từ gối đến mặt phẳng B (mm)", value=300.0)
    R_A = st.number_input("🛠️ Bán kính add mass mặt phẳng A (mm)", value=80.0)
    R_B = st.number_input("🛠️ Bán kính add mass mặt phẳng B (mm)", value=80.0)
    angle_step = st.number_input("🔄 Bội số góc chia vị trí add mass (°)", value=18, step=1)

with col2:
    mA = st.number_input("💣 Khối lượng mất cân bằng mặt phẳng A (g)", value=1.5)
    angleA = st.number_input("📐 Góc mất cân bằng mặt A (°)", value=30.0)
    mB = st.number_input("💣 Khối lượng mất cân bằng mặt phẳng B (g)", value=1.2)
    angleB = st.number_input("📐 Góc mất cân bằng mặt B (°)", value=270.0)
    offset_angle = st.number_input("🎯 Offset góc hiệu chỉnh thực tế (°)", value=0.0)

momentA = mA * R_A
momentB = mB * R_B

st.markdown("### ⚡ Tính toán moment cân bằng cần add")

angleA_rad = np.deg2rad(angleA)
angleB_rad = np.deg2rad(angleB)

Mx = momentA * np.cos(angleA_rad) + momentB * np.cos(angleB_rad)
My = momentA * np.sin(angleA_rad) + momentB * np.sin(angleB_rad)

A_matrix = np.array([[dA, dB],
                     [1, 1]])
b_vector = np.array([-My, -Mx])

try:
    solution = np.linalg.solve(A_matrix, b_vector)
    FA = solution[0]
    FB = solution[1]
    st.success(f"🔧 Kết quả moment cần add: FA = {FA:.2f} g.mm, FB = {FB:.2f} g.mm")
except np.linalg.LinAlgError:
    st.error("❌ Lỗi: Không giải được hệ phương trình (kiểm tra lại dA, dB)")

st.markdown("### 🛠️ Phân tách vector thành các thành phần mass theo bội số góc")

fixed_mass = st.number_input("⚙️ Fixed mass mỗi cục add (g)", value=0.45)
max_vectors = st.number_input("🔢 Giới hạn số vector tối đa (0 = không giới hạn)", value=0, step=1)

def split_vector(moment, radius, angle_step, fixed_mass):
    vectors = []
    remaining_moment = moment
    angle_list = np.arange(0, 360, angle_step)
    fixed_moment = fixed_mass * radius
    count = 0
    while abs(remaining_moment) >= fixed_moment and (max_vectors == 0 or count < max_vectors):
        angle_idx = count % len(angle_list)
        angle = angle_list[angle_idx]
        vectors.append((fixed_mass, angle))
        add_moment = fixed_mass * radius
        remaining_moment -= np.sign(moment) * add_moment
        count += 1
    if abs(remaining_moment) > 0.01:
        adaptive_mass = abs(remaining_moment) / radius
        if adaptive_mass > 0:
            adaptive_angle = angle_list[count % len(angle_list)]
            vectors.append((adaptive_mass, adaptive_angle))
    return vectors

vectors_A = split_vector(FA, R_A, angle_step, fixed_mass)
vectors_B = split_vector(FB, R_B, angle_step, fixed_mass)

st.write("### ✅ Thành phần vector mặt A:")
st.write(pd.DataFrame(vectors_A, columns=["Mass (g)", "Angle (°)"]))

st.write("### ✅ Thành phần vector mặt B:")
st.write(pd.DataFrame(vectors_B, columns=["Mass (g)", "Angle (°)"]))

st.markdown("### 📊 Biểu đồ vector các thành phần add mass (có offset góc)")

def plot_vectors(vectors, title, offset, radius):
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={'projection': 'polar'})
    for mass, angle in vectors:
        angle_rad = np.deg2rad(angle + offset)
        moment = mass * radius
        ax.arrow(angle_rad, 0, 0, moment,
                 width=0.02, color='b', alpha=0.7)
    ax.set_title(title)
    return fig

col_plot1, col_plot2 = st.columns(2)
with col_plot1:
    st.pyplot(plot_vectors(vectors_A, "Vector Add Mass - Plane A", offset_angle, R_A))
with col_plot2:
    st.pyplot(plot_vectors(vectors_B, "Vector Add Mass - Plane B", offset_angle, R_B))

df_A = pd.DataFrame(vectors_A, columns=["Mass (g)", "Angle (°)"])
df_B = pd.DataFrame(vectors_B, columns=["Mass (g)", "Angle (°)"])
df_export = pd.concat([df_A.assign(Plane="A"), df_B.assign(Plane="B")])

csv = df_export.to_csv(index=False).encode('utf-8')
st.download_button(
    label="💾 Download CSV Result",
    data=csv,
    file_name='balancing_result.csv',
    mime='text/csv'
)
