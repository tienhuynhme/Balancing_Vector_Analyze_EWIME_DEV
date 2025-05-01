# ===============================
# 📦 Import Libraries
# ===============================
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# 🏗️ App Layout & Title
# ===============================
st.set_page_config(layout="wide")
st.title("⚙️ Two-Plane Dynamic Balancing Tool (Phase Aligned, Coupled, Adaptive)")

st.markdown("### 🧾 Input Parameters")

# ===============================
# 📥 User Input
# ===============================
col1, col2 = st.columns(2)

with col1:
    dA = st.number_input("📏 Khoảng cách từ gối đến mặt phẳng A (mm)", value=25.5)
    dB = st.number_input("📏 Khoảng cách từ gối đến mặt phẳng B (mm)", value=9.3)
    rA_add = st.number_input("🛠️ Bán kính vị trí add mass mặt A (mm)", value=67.5)
    rB_add = st.number_input("🛠️ Bán kính vị trí add mass mặt B (mm)", value=67.5)
    angle_step = st.number_input("🔄 Bội số góc chia vị trí add mass (°)", value=18, step=1)

with col2:
    mA = st.number_input("💣 Khối lượng mất cân bằng mặt A (g)", value=3.47)
    rA_ub = st.number_input("📏 Bán kính vị trí mất cân bằng mặt A (mm)", value=74)
    angleA = st.number_input("📐 Góc mất cân bằng mặt A (°)", value=94.0)
    mB = st.number_input("💣 Khối lượng mất cân bằng mặt B (g)", value=0.45)
    rB_ub = st.number_input("📏 Bán kính vị trí mất cân bằng mặt B (mm)", value=74)
    angleB = st.number_input("📐 Góc mất cân bằng mặt B (°)", value=291.0)
    offset_angle = st.number_input("🎯 Offset góc hiệu chỉnh thực tế (°)", value=0.0)

fixed_mass = st.number_input("⚙️ Fixed mass mỗi cục add (g)", value=0.28)
max_vectors = st.number_input("🔢 Giới hạn số vector tối đa (0 = không giới hạn)", value=0, step=1)
# ===============================
# 💡 Tính moment mất cân bằng từng mặt
# ===============================
momentA = mA * rA_ub
momentB = mB * rB_ub

# Tính vector unbalance theo X, Y
Ux_A = momentA * np.cos(np.deg2rad(angleA))
Uy_A = momentA * np.sin(np.deg2rad(angleA))
Ux_B = momentB * np.cos(np.deg2rad(angleB))
Uy_B = momentB * np.sin(np.deg2rad(angleB))

# Tổng vector mất cân bằng hệ
Mx_total = Ux_A + Ux_B
My_total = Uy_A + Uy_B

# ===============================
# 🧠 Giải hệ Coupling Torque + Force
# ===============================
A_matrix = np.array([
    [dA, dB],  # torque: dA * FA + dB * FB = -My
    [1, 1]     # force:  FA + FB = -Mx
])

b_vector = np.array([
    -My_total,
    -Mx_total
])

try:
    FA, FB = np.linalg.solve(A_matrix, b_vector)
    st.success(f"🔧 Kết quả moment cần add: FA = {FA:.2f} g.mm, FB = {FB:.2f} g.mm")
except np.linalg.LinAlgError:
    st.error("❌ Lỗi: Không giải được hệ phương trình (kiểm tra lại dA, dB)")
    # ===============================
# 🧭 Align phase riêng từng mặt phẳng
# ===============================
def calc_phase(moment):
    angle = (np.rad2deg(np.arctan2(moment, 0)) + 360) % 360
    return (angle + 180 + offset_angle) % 360

phase_A = calc_phase(FA)
phase_B = calc_phase(FB)

st.markdown(f"### 🎯 Góc align phase mặt A: {phase_A:.2f}°")
st.markdown(f"### 🎯 Góc align phase mặt B: {phase_B:.2f}°")

# ===============================
# 🛠️ Hàm phân tách vector add mass
# ===============================
def split_vector(moment, radius, angle_step, fixed_mass, max_vectors, target_angle):
    vectors = []
    remaining_moment = moment
    angle_list = np.arange(0, 360, angle_step)
    fixed_moment = fixed_mass * radius
    count = 0

    angle_list = sorted(angle_list, key=lambda x: abs((x - target_angle + 180) % 360 - 180))  # Gần hướng target nhất

    while abs(remaining_moment) >= fixed_moment and (max_vectors == 0 or count < max_vectors):
        angle = angle_list[count % len(angle_list)]
        vectors.append((fixed_mass, angle))
        remaining_moment -= np.sign(moment) * fixed_moment
        count += 1

    if abs(remaining_moment) > 0.01:
        adaptive_mass = abs(remaining_moment) / radius
        adaptive_angle = angle_list[count % len(angle_list)]
        vectors.append((adaptive_mass, adaptive_angle))

    return vectors

# ===============================
# 🧩 Phân tách vector từng mặt
# ===============================
vectors_A = split_vector(FA, rA_add, angle_step, fixed_mass, max_vectors, phase_A)
vectors_B = split_vector(FB, rB_add, angle_step, fixed_mass, max_vectors, phase_B)

st.write("### ✅ Thành phần vector mặt A (Add Mass đề xuất):")
st.write(pd.DataFrame(vectors_A, columns=["Mass (g)", "Angle (°)"]))

st.write("### ✅ Thành phần vector mặt B (Add Mass đề xuất):")
st.write(pd.DataFrame(vectors_B, columns=["Mass (g)", "Angle (°)"]))
# ===============================
# 📉 Tính Residual tổng hệ sau khi add mass
# ===============================
def calc_vector_sum(vectors, radius):
    total_x = sum([mass * radius * np.cos(np.deg2rad(angle)) for mass, angle in vectors])
    total_y = sum([mass * radius * np.sin(np.deg2rad(angle)) for mass, angle in vectors])
    return total_x, total_y

add_Mx_A, add_My_A = calc_vector_sum(vectors_A, rA_add)
add_Mx_B, add_My_B = calc_vector_sum(vectors_B, rB_add)

# Tổng residual hệ
residual_Mx = Mx_total + add_Mx_A + add_Mx_B
residual_My = My_total + add_My_A + add_My_B
residual_magnitude = np.sqrt(residual_Mx**2 + residual_My**2)
residual_angle = (np.rad2deg(np.arctan2(residual_My, residual_Mx))) % 360

st.markdown(f"### ⚠️ Residual moment sau khi add mass: {residual_magnitude:.2f} g.mm @ {residual_angle:.2f}°")

# ===============================
# 📊 Biểu đồ vector
# ===============================
st.markdown("### 📈 Biểu đồ vector add mass và residual")

def plot_vectors(vectors, title, offset, radius, residual=None):
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={'projection': 'polar'})
    for mass, angle in vectors:
        angle_rad = np.deg2rad(angle + offset)
        moment = mass * radius
        ax.arrow(angle_rad, 0, 0, moment, width=0.02, color='b', alpha=0.7)
    if residual:
        res_angle_rad = np.deg2rad(residual[1] + offset)
        ax.arrow(res_angle_rad, 0, 0, residual[0], width=0.03, color='r', alpha=0.9, label='Residual')
        ax.legend(loc='upper right')
    ax.set_title(title)
    return fig

col_plot1, col_plot2 = st.columns(2)
with col_plot1:
    st.pyplot(plot_vectors(vectors_A, "Vector Add Mass - Plane A", offset_angle, rA_add, (residual_magnitude, residual_angle)))
with col_plot2:
    st.pyplot(plot_vectors(vectors_B, "Vector Add Mass - Plane B", offset_angle, rB_add, (residual_magnitude, residual_angle)))

# ===============================
# 💾 Xuất CSV kết quả
# ===============================
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
