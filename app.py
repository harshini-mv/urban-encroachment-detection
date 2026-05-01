import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# ------------------ UI ------------------
st.title("🛰️ Satellite-Based Urban Encroachment Detection System")

st.markdown("""
This system analyzes BEFORE and AFTER satellite images to detect urban expansion 
and possible encroachment using image processing techniques.
""")

# Sidebar settings
st.sidebar.title("Settings")
threshold = st.sidebar.slider("Change Sensitivity", 10, 100, 30)

# Upload files
before_file = st.file_uploader("Upload BEFORE Image", type=["jpg", "png"], key="before")
after_file = st.file_uploader("Upload AFTER Image", type=["jpg", "png"], key="after")

# ------------------ PROCESS ------------------
if before_file and after_file:

    before_img = Image.open(before_file)
    after_img = Image.open(after_file)

    # Display images
    st.subheader("📷 Input Images")
    col1, col2 = st.columns(2)

    col1.image(before_img, caption="Before", width="stretch")
    col2.image(after_img, caption="After", width="stretch")

    # ------------------ Comparison ------------------
    st.subheader("🆚 Before vs After Comparison")

    col3, col4 = st.columns(2)
    with col3:
        st.image(before_img, caption="Before", width="stretch")
    with col4:
        st.image(after_img, caption="After", width="stretch")

    # Convert to grayscale
    before = cv2.cvtColor(np.array(before_img), cv2.COLOR_RGB2GRAY)
    after = cv2.cvtColor(np.array(after_img), cv2.COLOR_RGB2GRAY)

    # Resize to match
    after = cv2.resize(after, (before.shape[1], before.shape[0]))

    # Reduce noise
    before = cv2.GaussianBlur(before, (5, 5), 0)
    after = cv2.GaussianBlur(after, (5, 5), 0)

    # Difference
    diff = cv2.absdiff(before, after)

    # Threshold
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw output
    output = cv2.cvtColor(before, cv2.COLOR_GRAY2BGR)

    change_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 800:  # noise filter
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
            change_area += area

    total_area = before.shape[0] * before.shape[1]
    change_percent = (change_area / total_area) * 100

    # ------------------ OUTPUT ------------------
    st.subheader("🔍 Detected Changes")
    st.image(output, caption="Highlighted Changes", width="stretch")

    # Download button
    _, buffer = cv2.imencode('.png', output)
    img_bytes = buffer.tobytes()

    st.download_button(
        label="📥 Download Result Image",
        data=img_bytes,
        file_name="encroachment_result.png",
        mime="image/png"
    )

    # ------------------ ANALYSIS ------------------
    st.subheader("📊 Analysis Report")

    st.metric("Change Percentage", f"{change_percent:.2f}%")
    st.metric("Affected Area (pixels)", int(change_area))

    if change_percent < 5:
        risk = "🟢 Low"
    elif change_percent < 15:
        risk = "🟡 Medium"
    else:
        risk = "🔴 High"

    st.write(f"### Encroachment Risk Level: {risk}")

    st.info("This is a prototype system for detecting urban expansion and encroachment using satellite imagery.")
    
# Run the app with: python -m streamlit run app.py