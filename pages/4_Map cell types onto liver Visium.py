
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
st.markdown("# Liver spatial cell types")
st.sidebar.header("Liver spatial cell types")

if 'show_image_2' not in st.session_state:
    st.session_state.show_image_2 = False

# Button to toggle the image display state
if st.button('Show Backgrounds'):
    # Toggle the state
    st.session_state.show_image_2 = not st.session_state.show_image_2

# If the state is True, display the image and caption
if st.session_state.show_image_2:
    st.write(
        """Why liver study should conside :green[spatial cell types]?   
        A liver lobule is the basic structural unit of the liver.   
        It consists of plates of hepatocytes (liver cells) radiating out from a central vein.   
        The periphery of each lobule contains portal triads, which consist of a portal vein, a hepatic artery, and a bile duct.  
        Blood flows into the lobules through the portal vein and hepatic artery, filters through the hepatocytes where nutrients are absorbed and toxins are metabolized, and then collects in the central vein to be transported back into general circulation.  

    """
    )
    st.image('./liver_1.png')
    st.caption('Image from https://www.frontiersin.org/articles/10.3389/fbioe.2022.845360/full')

animation_demo()

# show_code(animation_demo)
