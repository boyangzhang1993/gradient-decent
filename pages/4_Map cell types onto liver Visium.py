
import streamlit as st
import streamlit.components.v1 as components





def animation_demo() -> None:

    # Read the HTML file
    with open('./interactive_tangram_subset.html', 'r') as f:
        html_content = f.read()

    # Display the HTML content
    components.html(html_content, width=700, height=400)
    
    with open('./interactive_KIM_LIVER_CANCER_POOR_SURVIVAL_UP_subset.html', 'r') as f:
        html_content_gene_set = f.read()

    # Display the HTML content
    components.html(html_content_gene_set, width=700, height=400)
    
    with open('./interactive_YAMASHITA_LIVER_CANCER_STEM_CELL_UP_subset.html', 'r') as f:
        html_content_gene_set_2 = f.read()

    # Display the HTML content
    components.html(html_content_gene_set_2, width=700, height=400)


st.set_page_config(page_title="Liver spatial cell types", page_icon="📹")
st.markdown("# Animation Demo")
st.sidebar.header("Animation Demo")
st.write(
    """This app shows how you can use Streamlit to build cool animations.
It displays an animated fractal based on the the Julia Set. Use the slider
to tune different parameters."""
)

animation_demo()

# show_code(animation_demo)
