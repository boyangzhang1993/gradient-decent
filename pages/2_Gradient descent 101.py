

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from typing import Union, List
import copy
import plotly.graph_objects as go


def setup_session_state():
    if 'data' not in st.session_state:
        st.session_state['data'] = pd.DataFrame({
            "1 - Cosine similarity": [],
            "b": [],
            "color": [],
        })
    if 'data_slope' not in st.session_state:
        st.session_state['data_slope'] = pd.DataFrame({
            "x": [],
            "y": []
        })
    setup_legend_names() 

def setup_legend_names():
    if 'data' in st.session_state:
        st.session_state.data['legend'] = st.session_state.data['color'].replace({
            'blue': 'Blue: Exhausted Search',
            'red': 'Iteration'
        })

setup_legend_names() 


def simulate_linear_data(slope, point, x_range=(0, 10),num_points=10):
    '''
    Simulates data points around a linear line defined by a given slope and a point on the line.
    
    Parameters:
    - slope (float): The slope of the line.
    - point (tuple): A point (x0, y0) on the line.
    - std_dev (float): Standard deviation of the Gaussian noise to add to the y values.
    - x_range (tuple): The range (start, end) of x values.
    - num_points (int): Number of data points to generate.
    
    Returns:
    - x_values (numpy.ndarray): Generated x values.
    - y_values (numpy.ndarray): Generated y values with added noise.
    '''
    x0, y0 = point
    c = y0 - slope * x0  # Calculate the y-intercept from the point and slope

    # Generate linearly spaced x values
    x_values = np.linspace(x0-10, x0+10, num_points)
    
    # Calculate corresponding y values using the linear equation
    y_values = slope * x_values + c

    
    return x_values, y_values



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
    if 'slope_data' not in st.session_state:
        st.session_state['slope_data'] = None
    st.markdown("""
                # Gradient Descent

                Gradient Descent is an optimization algorithm used for minimizing a function by iteratively moving in the direction of steepest descent as defined by the negative of the gradient. It is commonly used in machine learning and deep learning for training models.

                ## Concept

                The main idea behind gradient descent is to update the parameters of the model iteratively to minimize the loss function. The steps involved are:

                1. **Initialize Parameters**: Start with initial values for the parameters that will be optimized.

                2. **Compute Gradient**: Calculate the gradient of the loss function with respect to each parameter. The gradient is a vector of partial derivatives, where each element represents the change in the loss function with respect to a parameter.

                3. **Update Parameters**: Adjust the parameters in the opposite direction of the gradient:

                $$
                \\theta = \\theta - \\eta \\cdot \\nabla_\\theta J(\\theta)
                $$

                Where:
                - $\\theta$ represents the parameters.
                - $\\eta$ is the learning rate, a scalar that determines the step size during the minimization process.
                - $\\nabla_\\theta J(\\theta)$ is the gradient of the loss function $J$ with respect to the parameters $\\theta$.

                4. **Repeat**: Repeat the process until the loss function converges to a minimum or a predetermined number of iterations is reached.
                """, 
                unsafe_allow_html=True)
    
    


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
    b_values_test = np.arange(-10, 10, 1)
    for b_value in b_values_test:

        M_matrix = np.array([[params_gradients['a'], 
                            b_value, 
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
        # if cos_value > cos_target:
        #     break

        # Capture current values
        a_values.append(params_gradients['a'])
        b_values.append(b_value)
        c_values.append(params_gradients['c'])
        cos_values.append(cos_value)
    

    df_countour = df = pd.DataFrame({
        'b': b_values,
        'cosine_similarity': [v[0][0] for v in cos_values],
        
    })
    
    # st.dataframe(df_countour.head())

    if 'data' not in st.session_state:
        # Start with an initial dataset
        st.session_state.data = pd.DataFrame({
            "1 - Cosine similarity": 1- df_countour['cosine_similarity'],
            "b": df_countour['b'],
            "color": ['blue']*len(b_values),
        })

        
        # st.session_state.slope_data = pd.DataFrame({
        #     "1 - Cosine similarity": 1- df_countour['cosine_similarity'],
        #     "b": df_countour['b'],
        #     "color": ['blue']*len(b_values),
        # })
        
    # if 'slope_data' not in st.session_state:
    #     # Start with an initial dataset
        
        


    # Plot the current data using Plotly
    # fig = px.line(st.session_state.data, x='b', y='1 - Cosine similarity', 
    #                  color='color', 
    #                  title='1 - Cosine Similarity with B Value', 
    #                  color_discrete_map={'blue': 'blue', 'red': 'red'}
    #                  )
    
    # fig.update_layout(
    #     title_font=dict(size=24, family='Arial Bold'),  # Make title font bigger and use a bold font if available
    #     font=dict(size=30, family='Arial Bold')  # Apply this font style to all text in the chart
    # )
    # st.plotly_chart(fig, use_container_width=True)
    
    def plot_data(b_start=None, cosine_point_1=None, gradient_approx=None):
        # Use Plotly Express for the main line plot
        setup_session_state()
        fig = px.line(st.session_state.data, x='b', y='1 - Cosine similarity',
                    color='legend',  # Use the 'legend' column for color grouping
                    title='1 - Cosine Similarity with B Value',
                    color_discrete_map={
                        'Blue: Exhausted Search': 'blue',  # Map the custom name to a color
                        'Iteration': 'red'
                    })

        # Add a scatter plot layer for the new data points only
        # new_data = st.session_state.data[st.session_state.data['color'] == 'red']
        new_data = st.session_state.data[st.session_state.data['legend'] == 'Iteration']
        if not new_data.empty:
            fig.add_trace(go.Scatter(x=new_data['b'], y=new_data['1 - Cosine similarity'],
                                    mode='markers', marker=dict(color='red', size=10),
                                    name='Iteration'))
        slop_data = st.session_state.data_slope
        if slop_data is not None:
            # Define start and end points for the line
        
            
            fig.add_trace(go.Scatter(x=slop_data['x'], y=slop_data['y'],
                                    mode='lines', 
                                    line=dict(color='green', width=4),
                                    name='Gradient Direction'))

        fig.update_layout(
            title_font=dict(size=24, family='Arial Bold'),
            font=dict(size=30, family='Arial Bold')
        )
        # st.plotly_chart(fig, use_container_width=True)
        
        return fig
    fig = plot_data()
    st.plotly_chart(fig, use_container_width=True) 

    b_start = 10
    
    # def update_iteration(b: float=b_start, params_gradients:dict=params_gradients,
    #                      )
    def update_b_iteration(simulate_spatial_data=simulate_spatial_data, params_gradients=params_gradients, b_start=b_start):
        M_matrix = np.array([[params_gradients['a'], 
                            b_start, 
                            params_gradients['c']]])
        min_vals = M_matrix.min(axis=1)
        max_vals = M_matrix.max(axis=1)

        M_matrix = (M_matrix - min_vals) / (max_vals - min_vals)
        params_gradients['a'] = M_matrix[0][0]
        params_gradients['b'] = M_matrix[0][1]
        params_gradients['c'] = M_matrix[0][2]

        prediction_genes = M_matrix.dot(single_cell_df)

        cos_value = cosine_similarity_matrix_formal(prediction_genes, simulate_spatial_data)
        params_gradient_test = {
            'arg_choose': 'b',
            'a':params_gradients['a'],
            'b':params_gradients['b'],
            'c':params_gradients['c'],
            'X':single_cell_df,
            'Y':simulate_spatial_data,
        }
        gradient_approx = derivative_approx(**params_gradient_test)
        return cos_value, gradient_approx
    cos_value, gradient_approx = update_b_iteration(simulate_spatial_data, params_gradients, b_start)
    
    if 'b_start' not in st.session_state:
        st.session_state.b_start = 10 

    # Button to add a new data point
    if st.button('Add one iteration'):
        # Generate a new data point and append it to the existing DataFrame
        st.write(f'Current b value is {st.session_state.b_start} gradient is {gradient_approx}, ')
        st.write(f'The new b value is {st.session_state.b_start} - {gradient_approx} = {st.session_state.b_start - gradient_approx}')
        
        new_data_point = pd.DataFrame({
            "b": [st.session_state.b_start],
            "1 - Cosine similarity": 1 - cos_value[0][0],
            "color": ['red'],
        })
        cos_value, gradient_approx = update_b_iteration(simulate_spatial_data, params_gradients, st.session_state.b_start)

        st.session_state.b_start -= 0.2 if gradient_approx > 0 else -0.2
        
        # st.write(f'{st.session_state.b_start}, {gradient_approx}')
        
        # new_data_point = pd.DataFrame({
        #     "b": [4],  # New b value
        #     "1 - Cosine similarity": [0.2],  # New cosine similarity value
        #     "color": ['red']  # Highlight new data points in red
        # })
        # Update the session state with the new data point
        st.session_state.data = pd.concat([st.session_state.data, new_data_point], ignore_index=True)
        x_vals, y_vals = simulate_linear_data(slope=gradient_approx*10000, point=(st.session_state.b_start, 
                                                                            1 - cos_value[0][0]),)

        st.session_state.data_slope = pd.DataFrame(
            
            {
                
                'x':x_vals,
                'y':y_vals
            }
        )
        fig = plot_data(b_start=st.session_state.b_start, 
                        cosine_point_1=1 - cos_value[0][0],
                        gradient_approx=gradient_approx)  # Update the plot data
        # st.plotly_chart(fig, use_container_width=True) 


    
data_frame_demo()

# show_code(data_frame_demo)
