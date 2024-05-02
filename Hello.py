

import streamlit as st
from streamlit.logger import get_logger
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from typing import Union, List
import time
import streamlit.components.v1 as components


LOGGER = get_logger(__name__)

# https://i.imgflip.com/8oo1fb.jpg

def show_image_in_expander(url='https://i.imgflip.com/8oo1fb.jpg'):
    with st.expander("Click to view the image"):
        st.image(url, caption="Here's your image!")
        
def show_temporary_image(url='https://i.imgflip.com/8oo1fb.jpg', timeout=5):
    # Display the image using the URL
    st.image(url, )
    
    # Wait for 'timeout' seconds
    # time.sleep(timeout)
    
    # Clear the previous output (the image)
    # st.experimental_rerun()
    
def cosine_similarity_matrix_formal(X:List[float], Y:List[float]) -> np.ndarray:
    x = np.array(X)
    y = np.array(Y)
    x_norm = np.sqrt(np.sum(x**2))
    y_norm = np.sqrt(np.sum(y**2, axis=1, keepdims=True))
    return np.dot(x, y.T) / (x_norm * y_norm)


single_cell_df = pd.DataFrame({
    'gene_1': [1, 0, 0],
    'gene_2': [2, 0, 0],
    'gene_3': [3, 0, 0],
    'gene_4': [0, 1, 0],
    'gene_5': [0, 2, 0],
    'gene_6': [0, 3, 0],
    'gene_7': [0, 0, 1],
    'gene_8': [0, 0, 2],
    'gene_9': [0, 0, 3],
    },
    index=['cell_A', 'cell_B', 'cell_C'],
)
def simulate_tangram_prediction_groundtruth(
    classic_example: bool = False,
    single_cell_df: pd.DataFrame = single_cell_df,
    # single_cell_adata: AnnData = single_cell_adata,
    seed: Union[None, int] = None,
    size_spot: int = 3,
):

    if seed:
        np.random.seed(seed)

    l_spot = ['spot_' + str(i) for i in range(size_spot)]
    simulate_spatial = pd.DataFrame(
        np.random.rand(size_spot, 3),  # Random a 3x3 matrix
        columns=['cell_A', 'cell_B', 'cell_C'],
        index=l_spot,
    )

    if classic_example:
        simulate_spatial = pd.DataFrame(
            [[0.5, 0, 0.5,]],  # Random a 3x3 matrix
            columns=['cell_A', 'cell_B', 'cell_C'],
            index=['spot_1'],
        )

    single_cell_df = single_cell_df.reindex(simulate_spatial.columns)



    simulate_spatial_data = single_cell_df.T.dot(simulate_spatial.T).T


    matrix_groundtruth = round(simulate_spatial, 5)



    return matrix_groundtruth, simulate_spatial_data


def run():
    

    st.set_page_config(
        page_title="Intro of Deep Learning for Visium Data",
        page_icon="👋",
    )

    st.write("# Intro of Deep Learning for Visium Data👋")

    st.sidebar.success("Select a page above.")
    st.caption('Author: Boyang Zhang from Stanly Ng lab at UCI')

    st.markdown(
        """
       
        
        Single-cell RNA and Visium data are two types of datasets used in the field of genomics, specifically in the study of biological tissues at a high resolution. Here's a brief introduction:
    """
    )
    
    st.subheader('Single Cell RNA Data')

    # st.image('./sc_1.png')
    # st.caption('Adapt from Ramachandran, P., Matchett, K.P., Dobie, R. et al. https://doi.org/10.1038/s41575-020-0304-x')
    
    if 'show_image' not in st.session_state:
        st.session_state.show_image = False

    # Button to toggle the image display state
    if st.button('Show Single Cell Backgrounds'):
        # Toggle the state
        st.session_state.show_image = not st.session_state.show_image

    # If the state is True, display the image and caption
    if st.session_state.show_image:
        st.image('./sc_1.png')
        st.caption('Adapted from Ramachandran, P., Matchett, K.P., Dobie, R. et al. https://doi.org/10.1038/s41575-020-0304-x')

    st.subheader('This is a Simulated Single Cell RNA Data')
    st.dataframe(data=single_cell_df)

    ## 
    # show_temporary_image(url = 'https://cdn.10xgenomics.com/image/upload/f_auto,q_auto,w_680,h_510,c_limit/v1574196658/blog/singlecell-v.-bulk-image.png')

    matrix_groundtruth, simulate_spatial_data = simulate_tangram_prediction_groundtruth(classic_example=True,
                                                                                        size_spot=1,)
    st.subheader('Spatial Transcriptomics (Visium)')

    # st.image('./visium_1.png')
    # st.caption('image from http://research.libd.org/VistoSeg/')
    
    if 'show_image_2' not in st.session_state:
        st.session_state.show_image_2 = False

    # Button to toggle the image display state
    if st.button('Show Visium Backgrounds'):
        # Toggle the state
        st.session_state.show_image_2 = not st.session_state.show_image_2

    # If the state is True, display the image and caption
    if st.session_state.show_image_2:
        st.image('./visium_1.png')
        st.caption('image from http://research.libd.org/VistoSeg/')
    
    
    st.subheader('This is a Simulated Visium Data')
    
    if st.button('Show/Hide Grouth Truth'):
        # Toggle the state
        if 'show_df' not in st.session_state:
            st.session_state.show_df = True
        else:
            st.session_state.show_df = not st.session_state.show_df
    if st.session_state.get('show_df', False):
        st.dataframe(data=matrix_groundtruth)
        st.write("This ground truth ")
    # st.dataframe(data=matrix_groundtruth)
    st.write("So we use ground truth to simulate a Visium data")
    st.dataframe(data=simulate_spatial_data)
    # Prediction
    st.subheader('Integrating Single Cell and Visium Spatial Gene Expression Data', divider='rainbow')
    st.markdown("""From **single cell RNA** we can infer :red[cell types] but no :blue[spatial] context.  
                   From **Visium**, we can infer :red[spatial] info but no :blue[cell types].  
                   Can we **integrate** single cell and Visium to infer the :green[spatiality of cell types]?""")
    st.image('visium_2.png')
    st.subheader('This is the Prediction We are Using',)
    
    a = st.slider('Select value for a', min_value=-100, max_value=100, value=1)
    b = st.slider('Select value for b', min_value=-100, max_value=100, value=10)
    c = st.slider('Select value for c', min_value=-100, max_value=100, value=1)

    # Create the matrix using numpy
    
    M_matrix = np.array([[a, b, c]])
    min_vals = M_matrix.min(axis=1)
    max_vals = M_matrix.max(axis=1)

    M_matrix = (M_matrix - min_vals) / (max_vals - min_vals)
    # Display the matrix
    # st.write('Normalized Matrix M (proportion of cells):', )
    # st.
    st.write('#### Normalized a, b, c to represent cell proportion')
    st.dataframe(data=pd.DataFrame(M_matrix, 
                                   columns=['cell_A','cell_B','cell_C']),
                 hide_index=True)
    st.write('#### So according to cell proportion, below is the predicted gene profiles')
    prediction_genes = M_matrix.dot(single_cell_df)
    st.dataframe(data=pd.DataFrame(prediction_genes,
                                   columns=['gene_1','gene_2','gene_3','gene_4','gene_5','gene_6','gene_7','gene_8','gene_9'],
                                   index=['spot_1']),
                 hide_index=True)
    if st.button('Show/Hide Visium'):
        # Toggle the state
        if 'show_df' not in st.session_state:
            st.session_state.show_df = True
        else:
            st.session_state.show_df = not st.session_state.show_df
        if st.session_state.get('show_df', False):
            st.write("Visium data we got")
            st.dataframe(data=simulate_spatial_data)
            

    cosine_init = cosine_similarity_matrix_formal(prediction_genes, simulate_spatial_data)
    
    
    if cosine_init > 0.92:
        st.success(f'Match found ! Similarity is {cosine_init[0][0]}')
        st.toast('You did good job on Prediction!', icon='🎉')
        # mycode = "<a href='https://imgflip.com/i/8oo1fb'><img src='https://i.imgflip.com/8oo1fb.jpg' /></a><div><a href='https://imgflip.com/memegenerator'>from Imgflip Meme Generator</a></div>"
        # components.html(mycode, height=0, width=0)
        st.balloons()
        time.sleep(3)
        st.image('./success_1.png')

        # show_image_in_expander()
        # show_temporary_image()
    elif cosine_init > 0.5:
        st.info(f'Ok, try it again ! Similarity is {cosine_init[0][0]}')
        # st.snow()

    else:
        st.error(f'Well, we can do better ! Similarity is {cosine_init[0][0]}')
        
    
    
    st.caption(f'The similarity of prediction and truth (Cosine similiarity): {cosine_init[0][0]}')
    
    
    # 
    



if __name__ == "__main__":
    
    
    run()
