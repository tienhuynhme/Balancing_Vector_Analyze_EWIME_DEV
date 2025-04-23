
import streamlit as st
import numpy as np
import pandas as pd
import datetime
import io
import matplotlib.pyplot as plt

st.set_page_config(page_title="Cân Bằng Động Hybrid - 2 Mặt Phẳng", layout="wide")
st.title("⚙️ Cân Bằng Động Hybrid (Giới hạn vector + Biểu đồ)")

if "history" not in st.session_state:
    st.session_state.history = []

def plot_vectors(components, offset_angle, label):
    plt.figure(figsize=(5, 5))
    plt.axhline(0, color="gray", linewidth=0.5)
    plt.axvline(0, color="gray", linewidth=0.5)
    for m, r, a, loai in components:
        angle_rad = np.radians(a)
        x = m * r * np.cos(angle_rad)
        y = m * r * np.sin(angle_rad)
        plt.arrow(0, 0, x, y, head_width=10, head_length=15, fc="blue", ec="blue", alpha=0.7)
    plt.title(f"Biểu đồ Vector - Mặt {label}")
    plt.xlabel("Moment X (g.mm)")
    plt.ylabel("Moment Y (g.mm)")
    plt.xlim(-500, 500)
    plt.ylim(-500, 500)
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    st.pyplot(plt)
    plt.close()

def process_plane(label):
    st.subheader(f"🔵 Mặt Phẳng {label}")

    col1, col2 = st.columns(2)
    with col1:
        m_input = st.number_input(f"Khối lượng mất cân bằng (g) - {label}", min_value=0.0, value=2.0, key=f"m_{label}")
        r_input = st.number_input(f"Bán kính vị trí mất cân bằng (mm) - {label}", min_value=1.0, value=70.0, key=f"r_{label}")
        theta = st.number_input(f"Góc mất cân bằng (°) - {label}", min_value=0.0, max_value=360.0, value=60.0, key=f"theta_{label}")
        offset_angle = st.number_input(f"Offset góc thao tác (°) - {label}", min_value=-180.0, max_value=180.0, value=18.0, key=f"offset_{label}")
    with col2:
        fixed_r = st.number_input(f"Bán kính gắn mass (mm) - {label}", min_value=1.0, value=50.0, key=f"fr_{label}")
        fixed_m = st.number_input(f"Khối lượng cố định mỗi vector (g) - {label}", min_value=0.01, value=0.5, key=f"fm_{label}")

    run = st.button(f"🚀 Phân Tách Vector - Mặt {label}")

    if run:
        M = m_input * r_input
        theta_rad = np.radians(theta)
        Mx_target = M * np.cos(theta_rad)
        My_target = M * np.sin(theta_rad)
        Mx, My = Mx_target, My_target
        Mi = fixed_m * fixed_r

        angle_step = 18
        angle_list = np.arange(0, 360, angle_step)
        sorted_angles = sorted(angle_list, key=lambda a: abs(a - theta) if abs(a - theta) <= 180 else 360 - abs(a - theta))
        valid_angles = np.radians(sorted_angles)

        components = []
        used_angles = set()

        # Tối đa 3 vector cố định
        for angle in valid_angles:
            if len(components) >= 3:
                break
            vx = Mi * np.cos(angle)
            vy = Mi * np.sin(angle)
            Mx -= vx
            My -= vy
            components.append((fixed_m, fixed_r, (np.degrees(angle) + offset_angle) % 360, "Cố định"))
            used_angles.add(angle)

        # Thêm 1 vector tùy biến cuối nếu còn sai số đáng kể
        remaining_moment = np.sqrt((Mx_target - Mx)**2 + (My_target - My)**2)
        if remaining_moment > 0.01:
            angle = np.arctan2(My_target - My, Mx_target - Mx)
            angle_deg = round(np.degrees(angle) / angle_step) * angle_step
            if angle_deg % 360 not in [round(a) % 360 for _, _, a, _ in components]:
                m_last = remaining_moment / fixed_r
                components.append((m_last, fixed_r, (angle_deg + offset_angle) % 360, "Tùy biến"))

        df = pd.DataFrame(components, columns=["Khối lượng (g)", "Bán kính (mm)", "Góc (°)", "Loại"])
        st.dataframe(df)

        check_Mx = sum(m * r * np.cos(np.radians(a - offset_angle)) for m, r, a, _ in components)
        check_My = sum(m * r * np.sin(np.radians(a - offset_angle)) for m, r, a, _ in components)
        residual_moment = np.sqrt((Mx_target - check_Mx)**2 + (My_target - check_My)**2)

        st.write(f"📉 Sai số còn lại: **{residual_moment:.2f} g.mm**")

        plot_vectors(components, offset_angle, label)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.history.append({
            "Thời gian": timestamp,
            "Mặt phẳng": label,
            "Khối lượng đo (g)": m_input,
            "Bán kính đo (mm)": r_input,
            "Góc mất cân bằng (°)": theta,
            "Khối lượng vector (g)": fixed_m,
            "Bán kính add mass (mm)": fixed_r,
            "Offset góc": offset_angle,
            "Số vector sinh ra": len(components),
            "Sai số còn lại": round(residual_moment, 4)
        })

st.markdown("---")
colA, colB = st.columns(2)
with colA:
    process_plane("A")
with colB:
    process_plane("B")

if st.session_state.history:
    if st.button("📤 Tải tất cả kết quả test"):
        df_export = pd.DataFrame(st.session_state.history)
        csv_buffer = io.StringIO()
        df_export.to_csv(csv_buffer, index=False)
        st.download_button("📥 Nhấn để tải file CSV", csv_buffer.getvalue(), file_name="lich_su_test_2matphang.csv", mime="text/csv")
