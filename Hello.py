

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

# def plot_function(f, x_range, current_x):
#     x = np.linspace(x_range[0], x_range[1], 400)
#     y = eval(f)
#     plt.figure(figsize=(10, 5))
#     plt.plot(x, y, label=f'f(x) = {f}')
#     plt.scatter([current_x], [eval(f.replace('x', f'({current_x})'))], color='red')  # Plot current point
#     plt.title("Function Graph")
#     plt.xlabel("x")
#     plt.ylabel("f(x)")
#     plt.legend()
#     plt.grid(True)
#     plt.close()
#     return plt




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
        page_title="Hello",
        page_icon="👋",
    )

    st.write("# Welcome to Streamlit! 👋")

    st.sidebar.success("Select a demo above.")

    st.markdown(
        """
        Streamlit is an open-source app framework built specifically for
        Machine Learning and Data Science projects.
        **👈 Select a demo from the sidebar** to see some examples
        of what Streamlit can do!
    """
    )
    st.subheader('This is the Single Cell Data')
    st.dataframe(data=single_cell_df)

    ## 
    matrix_groundtruth, simulate_spatial_data = simulate_tangram_prediction_groundtruth(classic_example=True,
                                                                                        size_spot=1,)
    st.subheader('This is the Simulated Visium Data')
    
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
    st.subheader('This is a Prediction we are using', divider='rainbow')
    
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
    
    
    if cosine_init > 0.9:
        st.success(f'Match found ! Similarity is {cosine_init[0][0]}')
        st.toast('You did good job on Prediction!', icon='🎉')
        # mycode = "<a href='https://imgflip.com/i/8oo1fb'><img src='https://i.imgflip.com/8oo1fb.jpg' /></a><div><a href='https://imgflip.com/memegenerator'>from Imgflip Meme Generator</a></div>"
        # components.html(mycode, height=0, width=0)
        st.balloons()

        # show_image_in_expander()
        show_temporary_image()
    elif cosine_init > 0.5:
        st.info(f'Ok, try it again ! Similarity is {cosine_init[0][0]}')
        # st.snow()

    else:
        st.error(f'Well, we can do better ! Similarity is {cosine_init[0][0]}')
        
    
    
    st.caption(f'The similarity of prediction and truth (Cosine similiarity): {cosine_init[0][0]}')
    
    
    # 
    



if __name__ == "__main__":
    
    
    run()
