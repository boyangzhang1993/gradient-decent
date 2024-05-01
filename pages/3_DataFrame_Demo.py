

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from typing import Union, List
import copy
import plotly.graph_objects as go
import time

def cosine_similarity_matrix_formal(X:List[float], Y:List[float]) -> np.ndarray:
    x = np.array(X)
    y = np.array(Y)
    x_norm = np.sqrt(np.sum(x**2))
    y_norm = np.sqrt(np.sum(y**2, axis=1, keepdims=True))
    return np.dot(x, y.T) / (x_norm * y_norm)

def cosine_loss_autograd(a: float, 
                         b: float, 
                         c: float, Y
                         : np.ndarray, supply_x:Union[np.ndarray, None]=None) -> float:
    x = np.array([[a, b, c]])
    if supply_x is not None:
        M_matrix = np.array([[a, b, c]])
        x = np.array(M_matrix.dot(supply_x))

    similarity = cosine_similarity_matrix_formal(x, Y)
    return 1 - np.mean(similarity)

def derivative_approx(
    arg_choose: str = 'a', 
    const: float = 0.0001, 
    a: float = 1, 
    b: float = 1, 
    c: float = 1, 
    X: Union[np.ndarray, None] = None,
    Y: np.ndarray = np.array([[1, 1, 1]]
                             
                             )
) -> float:
    assert arg_choose in {'a', 'b', 'c'}

    params = {
        'a': a,
        'b': b,
        'c': c,
        'supply_x': X,
        'Y': Y,

    }

    params_add = copy.deepcopy(params)
    params_delete = copy.deepcopy(params)

    params_add[arg_choose] += const
    # print(params_add[arg_choose])

    params_delete[arg_choose] -= const
    # print(params_delete[arg_choose])
    gradient_approx = (cosine_loss_autograd(**params_add) - 
                       cosine_loss_autograd(**params_delete)) / (2*const)

    return gradient_approx

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


def data_frame_demo():

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    
    
    max_iterations = 100000 # To avoid infinite loop
    cos_target = 0.98 # Target cosine similarity
    learning_rate = 0.1 # Learning rate

    # Reset 

    a=1
    b=10
    c=-1
    matrix_groundtruth, simulate_spatial_data = simulate_tangram_prediction_groundtruth(classic_example=True,
                                                                                            size_spot=1,)

    l_params = ['a', 'b', 'c']
    params_gradients = {
        'arg_choose': 'a',
        'a':a,
        'b':b,
        'c':c,
        'X':single_cell_df,
        'Y':simulate_spatial_data,
        

    }
    a_values, b_values, c_values, cos_values = [], [], [], []
    grad_a, grad_b, grad_c = [], [], []
    # Start the loop
    

    for _ in range(max_iterations):
        
        # new_rows = last_rows[-1, :] + np.random.randn(1, 1).cumsum(axis=0)
        status_text.text("%i%% Complete" % _)


        M_matrix = np.array([[params_gradients['a'], 
                            params_gradients['b'], 
                            params_gradients['c']]])
        min_vals = M_matrix.min(axis=1)
        max_vals = M_matrix.max(axis=1)

        M_matrix = (M_matrix - min_vals) / (max_vals - min_vals)
        params_gradients['a'] = M_matrix[0][0]
        params_gradients['b'] = M_matrix[0][1]
        params_gradients['c'] = M_matrix[0][2]

        
        prediction_genes = M_matrix.dot(single_cell_df)

        cos_value = cosine_similarity_matrix_formal(prediction_genes, simulate_spatial_data)
        
        print(f'Cosine similarity: {cos_value}')
        if cos_value > cos_target:
            break

        # Capture current values
        a_values.append(params_gradients['a'])
        b_values.append(params_gradients['b'])
        c_values.append(params_gradients['c'])
        cos_values.append(cos_value)
        
        dict_gradients = {}
        
        max_gradient_iteration = float('-inf')
        for i, param in enumerate(l_params):
            params_gradient_test = {
                'arg_choose': param,
                'a':params_gradients['a'],
                'b':params_gradients['b'],
                'c':params_gradients['c'],
                'X':single_cell_df,
                'Y':simulate_spatial_data,
            }
            gradient_approx = derivative_approx(**params_gradient_test)

            if param == 'a':
                grad_a.append(gradient_approx)
            elif param == 'b':
                grad_b.append(gradient_approx)
            elif param == 'c':
                grad_c.append(gradient_approx)
            params_gradients[param] -= learning_rate * gradient_approx


            if gradient_approx > max_gradient_iteration:
                max_gradient_iteration = gradient_approx
                max_gradient_param = param
            dict_gradients[param] = gradient_approx
            
            
            

    df_countour = df = pd.DataFrame({
        'a': a_values,
        'b': b_values,
        'c': c_values,
        'grad_a': grad_a,
        'grad_b': grad_b,
        'grad_c': grad_c,
        'cosine_similarity': [v[0][0] for v in cos_values],
        
    })
    data = pd.DataFrame([[float(a),float(b),float(c),
                        #   float(grad_a[0]),
                        #   float(grad_b[0]),
                        #   float(grad_c[0]),
                        #   float([v[0][0] for v in cos_values][0]),
                          ]
                         
                         ],columns=["A", "B", "C", 
                                    # "Gradient_A", "Gradient_B", "Gradient_C", 'Cosine similarity'
                                    ])

    my_data_element = st.line_chart(data, color=['#1F77B4', 
                                                  '#D62728', 
                                                  '#E377C2'])
    for i, row in df.iterrows():
        a = float(row['a'])
        b = float(row['b'])
        c = float(row['c'])
        a_g = float(row['grad_a'])
        b_g = float(row['grad_b'])
        c_g = float(row['grad_c'])
        c_s = float(row['cosine_similarity'])
        
        
        # add_df = pd.DataFrame([[float(M_matrix[0][0]),float(M_matrix[0][0]),float(M_matrix[0][0])
        #                         ]
        #                        ], columns=(["A", "B", "C"]))
        # my_data_element.add_rows(add_df)
        # st.write(f'{a}')
        
        add_df = pd.DataFrame(np.array([[a,b,c,a_g,b_g,c_g,c_s]]), columns=["A", "B", "C", "Gradient_A", "Gradient_B", "Gradient_C", 'Cosine similarity'])
        my_data_element.add_rows(add_df)  # Correctly using add_rows on a dataframe element
        progress_bar.progress(_)
        # last_rows = new_rows
        time.sleep(0.05)
    data = pd.DataFrame([[
        # float(a),float(b),float(c),
                        float(grad_a[0]),
                        float(grad_b[0]),
                        float(grad_c[0]),
                        float([v[0][0] for v in cos_values][0]),
                        ]
                        
                        ],columns=[
                            # "A", "B", "C", 
                            "Gradient_A", "Gradient_B", "Gradient_C", 'Cosine similarity'])

    my_data_element = st.line_chart(data)
    for i, row in df.iterrows():
        a = float(row['a'])
        b = float(row['b'])
        c = float(row['c'])
        a_g = float(row['grad_a'])
        b_g = float(row['grad_b'])
        c_g = float(row['grad_c'])
        c_s = float(row['cosine_similarity'])
        
        
        # add_df = pd.DataFrame([[float(M_matrix[0][0]),float(M_matrix[0][0]),float(M_matrix[0][0])
        #                         ]
        #                        ], columns=(["A", "B", "C"]))
        # my_data_element.add_rows(add_df)
        # st.write(f'{a}')
        
        add_df = pd.DataFrame(np.array([[
            # a,b,c,
                                         a_g,b_g,c_g,c_s]]), columns=[
                                            #  "A", "B", "C", 
                                             "Gradient_A", "Gradient_B", "Gradient_C", 'Cosine similarity'])
        my_data_element.add_rows(add_df)  # Correctly using add_rows on a dataframe element
        progress_bar.progress(_)
        # last_rows = new_rows
        time.sleep(0.05)
        
        
        
    fig1 = go.Figure(data=[go.Scatter3d(
        x=df['c'],
        y=df['b'],
        z=df['cosine_similarity'],
        mode='markers',
        marker=dict(
            size=5,
            color=df['cosine_similarity'],  # set color to cosine_similarity
            colorscale='Viridis',  # choose a colorscale
            opacity=0.8
        )
    )])
    
    fig1.update_layout(
    title='3D Scatter Plot of Cosine Similarity',
    scene=dict(
        xaxis_title='c',
        yaxis_title='b',
        zaxis_title='Cosine Similarity'
    ),
    margin=dict(l=0, r=0, b=0, t=50)
    )
    
    fig2 = go.Figure(data=[go.Scatter3d(
        x=df['a'],
        y=df['b'],
        z=df['cosine_similarity'],
        mode='markers',
        marker=dict(
            size=5,
            color=df['cosine_similarity'],
            colorscale='Viridis',
            opacity=0.8
        )
    )])
    fig2.update_layout(
        title='3D Scatter Plot of Cosine Similarity',
        scene=dict(
            xaxis_title='a',
            yaxis_title='b',
            zaxis_title='Cosine Similarity'
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    # Streamlit layout for side-by-side plots
    col1, col2 = st.columns(2)

    # Displaying the figure in Streamlit
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)


data_frame_demo()





# show_code(data_frame_demo)
