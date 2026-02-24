---
name: enterprise-streamlit-component
description: Adds Streamlit UI components following enterprise standards. Use when building Streamlit pages, adding UI components, or when the user mentions Streamlit, st.session_state, or file uploads.
---

# Enterprise Streamlit Component

When adding a Streamlit UI component:

1. **Separate UI logic from business logic** — UI functions only in `app/ui/`
2. **Use `st.session_state` for all state management** — never global variables
3. **Validate all user inputs** before passing them downstream
4. **File uploads**: validate MIME type, enforce max size (configurable via env var)
5. **Show a spinner** for any operation > 500ms
6. **Display user-friendly error messages** — never expose stack traces
7. **Include a docstring** at the top of each page/component file describing its purpose
