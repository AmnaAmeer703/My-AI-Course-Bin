
    if question.strip():

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.spinner("Thinking..."):
            answer = ask_inventory_bot(
                question,
                inventory,
                source
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        st.rerun()

st.sidebar.success("✅ Sidebar reached")