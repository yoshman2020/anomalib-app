import streamlit as st


def disp_footer():
    """Display footer information."""
    print("disp_footer called")
    st.divider()
    st.caption(
        "© 2026 Your Company Name",
    )
    col1, col2 = st.columns(
        2, gap="xxsmall", vertical_alignment="center", width=650
    )
    with col1:
        st.caption(
            "This application uses third-party open-source software.",
        )
    with col2:
        # LICENSE読み込み
        with open("LICENSE", "rb") as f:
            data = f.read()
        st.download_button(
            label="LICENSE",
            data=data,
            file_name="LICENSE",
            mime="text/plain",
            type="tertiary",
        )
    print("disp_footer finished")
