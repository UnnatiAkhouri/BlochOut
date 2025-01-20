import plotly.graph_objects as go
import numpy as np
import streamlit as st

import base64

st.set_page_config(layout="wide")

st.markdown("""
    <style>
    .stSlider [data-baseweb=slider]{
        width: 25%;
    }
    </style>
    """,unsafe_allow_html=True)
def set_background(image_path):
    """
    Set a background image for the Streamlit app.
    :param image_path: Path to the background image file.
    """
    with open(image_path, "rb") as img_file:
        encoded_img = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_img}");
            background-size:contain ;
            background-position: 50%;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


set_background("/Users/jackiedipietro/Documents/GitHub/BlochOut/4.png")

st.title('BlochOut - Dual Qubits')

# Initialize session state for gate history if it doesn't exist
if 'gate_history1' not in st.session_state:
    st.session_state.gate_history1 = []
if 'gate_history2' not in st.session_state:
    st.session_state.gate_history2 = []


# Functions to convert quantum state to Bloch vector and calculate rho
def state_to_bloch_vector(state):
    alpha, beta = state
    x = 2 * np.real(alpha * np.conj(beta))
    y = 2 * np.imag(alpha * np.conj(beta))
    z = np.abs(alpha)**2 - np.abs(beta)**2
    return np.array([x, y, z])

P_x= np.array([[0, 1], [1, 0]])
P_y=np.array([[0, -1j], [1j, 0]])
P_z=np.array([[1, 0], [0, -1]])

def compute_pauli_expansion(state_vector):
    alpha, beta = state_vector
    expectation_x = np.real(alpha * np.conj(beta))
    expectation_y = np.imag(alpha * np.conj(beta))
    expectation_z = np.abs(alpha) ** 2 - np.abs(beta) ** 2
    rho = 0.5 * (np.eye(2) + expectation_x * np.array([[0, 1], [1, 0]]) +
                 expectation_y * np.array([[0, -1j], [1j, 0]]) +
                 expectation_z * np.array([[1, 0], [0, -1]]))
    return rho, expectation_x, expectation_y, expectation_z



# Function to generate a random quantum state
def generate_random_state():
    r = np.random.uniform(0, 1)
    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2 * np.pi)
    alpha = r * np.cos(theta / 2)
    beta = r * np.sin(theta / 2) * np.exp(1j * phi)
    return [alpha, beta]

# Gate functions
def apply_x_gate(state_vector):
    """Apply X gate to the given state vector."""
    x_gate = np.array([[0, 1], [1, 0]])
    return np.dot(x_gate, state_vector)

def apply_y_gate(state_vector):
    """Apply X gate to the given state vector."""
    y_gate = np.array([[0, -1j], [1j, 0]])
    return np.dot(y_gate, state_vector)

def apply_z_gate(state_vector):
    """Apply Z gate to the given state vector."""
    z_gate = np.array([[1, 0], [0, -1]])
    return np.dot(z_gate, state_vector)

def apply_h_gate(state_vector):
    """Apply H gate (Hadamard) to the given state vector."""
    h_gate = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
    return np.dot(h_gate, state_vector)

def apply_rotation_x_gate(state_vector,p):
    rx_matrix = np.array([[np.cos(p/2), -1j * np.sin(p/2)],
                     [- 1j *np.sin(p/2),  np.cos(p/2)]])
    return np.dot(rx_matrix, state_vector)

def apply_rotation_y_gate(state_vector,p):
    ry_matrix = np.array([[np.cos(p/2), -np.sin(p/2)],
                     [np.sin(p/2),  np.cos(p/2)]])
    return np.dot(ry_matrix, state_vector)

def apply_rotation_z_gate(state_vector,p):
    rz_matrix = np.array([[np.exp(-1j * p / 2), 0],
                     [0, np.exp(1j * p / 2)]])
    return np.dot(rz_matrix, state_vector)

def amplitude_damping_channel(state_vector, p):
    rho = compute_pauli_expansion(state_vector)[0]
    E0 = np.array([[1, 0], [0, np.sqrt(1 - p)]])
    E1 = np.array([[0, np.sqrt(p)], [0, 0]])
    new_rho = E0 @ rho @ E0.conj().T + E1 @ rho @ E1.conj().T
    return np.array([np.trace(new_rho @ P_x), np.trace(new_rho @ P_y), np.trace(new_rho @ P_z)])

def dephasing_channel(state_vector, p):
    rho = compute_pauli_expansion(state_vector)[0]
    E0 = np.sqrt(1 - p) * np.eye(2)
    E1 = np.sqrt(p) * np.array([[1, 0], [0, -1]])
    new_rho = E0 @ rho @ E0.conj().T + E1 @ rho @ E1.conj().T
    return np.array([np.trace(new_rho @ P_x), np.trace(new_rho @ P_y), np.trace(new_rho @ P_z)])




if 'initial_state_vector1' not in st.session_state:
    st.session_state.initial_state_vector1 = generate_random_state()

# Initialize the first and second state vectors
if 'state_vector1' not in st.session_state:
    #st.session_state.state_vector1 = generate_random_state()

    st.session_state.state_vector1 = st.session_state.initial_state_vector1

    initial_state_vector1 = st.session_state.initial_state_vector1

    # Apply gates to the initial state to get the final state
    final_state_vector1 = apply_z_gate(apply_z_gate(apply_y_gate(apply_x_gate(initial_state_vector1))))
    final_bloch_vector1 = state_to_bloch_vector(final_state_vector1)

if 'initial_state_vector2' not in st.session_state:
    st.session_state.initial_state_vector2 = generate_random_state()

if 'state_vector2' not in st.session_state:
    st.session_state.state_vector2 = st.session_state.initial_state_vector2

    initial_state_vector2 = st.session_state.initial_state_vector2

    # Apply gates to the initial state to get the final state
    final_state_vector2 = apply_h_gate(apply_z_gate(apply_y_gate(apply_x_gate(initial_state_vector2))))
    final_bloch_vector2 = state_to_bloch_vector(final_state_vector2)

# Function to plot Bloch sphere
def plot_bloch_sphere(bloch_vector, final_bloch, color, title):
    fig = go.Figure()
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    fig.add_surface(x=x, y=y, z=z, colorscale=color, opacity=0.6, showscale=False)

    fig.add_trace(go.Scatter3d(
        x=[bloch_vector[0]], y=[bloch_vector[1]], z=[bloch_vector[2]],
        mode='markers+text',
        marker=dict(size=5, color='black'),
        name='',
        text=['State'],
    ))

    fig.add_trace(go.Scatter3d(
        x=[0, bloch_vector[0]], y=[0, bloch_vector[1]], z=[0, bloch_vector[2]],
        mode='lines+text',
        line=dict(color='black', width=4),
        name=""
    ))

    fig.add_trace(go.Scatter3d(
        x=[final_bloch[0]], y=[final_bloch[1]], z=[final_bloch[2]],
        mode='markers+text',
        marker=dict(size=5, color='magenta'),
        name='',
        text=['Final State'],
    ))

    fig.add_trace(go.Scatter3d(
        x=[0, final_bloch[0]], y=[0, final_bloch[1]], z=[0, final_bloch[2]],
        mode='lines+text',
        line=dict(color='magenta', width=4),
        name=""
    ))

    #chappal_data = chappal_data(bloch_vector[0], bloch_vector[1], bloch_vector[2])

    #fig.add_trace(
        #go.Scatter3d(x=chappal_data[0], y=chappal_data[1],
                     #z=chappal_data[2],
                     #mode='lines', line=dict(color='black', width=4)))

    fig.update_layout(
        scene=
        dict(
            xaxis=dict(nticks=4, range=[-1, 1], showbackground=False, zeroline=False),
            yaxis=dict(nticks=4, range=[-1, 1], showbackground=False, zeroline=False),
            zaxis=dict(nticks=4, range=[-1, 1], showbackground=False, zeroline=False),
            bgcolor='rgba(0,0,0,0)',  # Transparent background for the entire 3D plot
        ),autosize=False, width=550, height=550,
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the entire figure
        margin=dict(l=0, r=0, b=0, t=0),
        #title=title
    )
    st.plotly_chart(fig)


# Function to update Bloch sphere and circuit
def update_system(bloch_vector, state_vector, gate_history, circuit_color, circuit_key):
    bloch_vector[:] = state_to_bloch_vector(state_vector)
    draw_circuit(gate_history, circuit_color, circuit_key)


# Function to draw quantum circuit
def draw_circuit(gate_history, color, circuit_key):

    max_gates = 5
    truncated_history = gate_history[-max_gates:]  # Keep only the last 5 gates
    num_gates = len(truncated_history)

    circuit_fig = go.Figure()
    circuit_fig.add_trace(go.Scatter(
        x=[0, num_gates + 1],
        y=[0, 0],
        mode="lines",
        line=dict(width=3, color="black"),
        name="Wire"
    ))

    for i, gate in enumerate(truncated_history):
        circuit_fig.add_trace(go.Scatter(
            x=[i + 1], y=[0],
            mode="markers+text",
            marker=dict(symbol="square", size=40, color=color),
            text=[gate],
            textposition='middle center',
            name=f"Gate {i + 1}"
        ))

    circuit_fig.update_layout(
        xaxis=dict(range=[0, max_gates + 1], zeroline=False, showticklabels=False),
        yaxis=dict(range=[-1, 1], zeroline=False, showticklabels=False),
        margin=dict(l=0, r=0, b=0, t=0),
        height=100,width=400,
        showlegend=False
    )
    st.plotly_chart(circuit_fig, key=circuit_key)

# Function to display state and rho
def display_state_and_rho(state_vector):
    alpha, beta = state_vector
    state_string = f"$|\psi> = ({alpha:.2f})|0> + ({beta:.2f})|1>$"
    rho, expectation_x, expectation_y, expectation_z = compute_pauli_expansion(state_vector)
    rho_string = f"ρ = 0.5 (I + {expectation_x:.2f}σx + {expectation_y:.2f}σy + {expectation_z:.2f}σz)"
    st.write(state_string)
    st.write(rho_string)



# Display Bloch spheres side by side
col1, col2,col3,col4 = st.columns([5,5,5,5])

with col2:
    st.header("Qubit 1")
    if 'initial_state_vector1' not in st.session_state:
        st.session_state.initial_state_vector1 = generate_random_state()

    # Initialize the first and second state vectors
    if 'state_vector1' not in st.session_state:
        # st.session_state.state_vector1 = generate_random_state()

        st.session_state.state_vector1 = st.session_state.initial_state_vector1

    initial_state_vector1 = st.session_state.initial_state_vector1

        # Apply gates to the initial state to get the final state
    final_state_vector1 = apply_z_gate(apply_z_gate(apply_y_gate(apply_x_gate(initial_state_vector1))))
    final_bloch_vector1 = state_to_bloch_vector(final_state_vector1)
    bloch_vector1 = state_to_bloch_vector(st.session_state.state_vector1)
    #final_bloch_vector1 = state_to_bloch_vector(final_state_vector1)

    plot_bloch_sphere(bloch_vector1, final_bloch_vector1, 'pinkyl', "Bloch Sphere 1")
    display_state_and_rho(st.session_state.state_vector1)

    st.write("Apply gate to Qubit 1:")
    if st.button('X Gate (Qubit 1)'):
        st.session_state.state_vector1 = apply_x_gate(st.session_state.state_vector1)
        st.session_state.gate_history1.append('X')
        #update_system(bloch_vector1, st.session_state.state_vector1, st.session_state.gate_history1, 'pink', 'circuit1')

    if st.button('Y Gate (Qubit 1)'):
        st.session_state.state_vector1 = apply_y_gate(st.session_state.state_vector1)
        st.session_state.gate_history1.append('Y')
        #update_system(bloch_vector1, st.session_state.state_vector1, st.session_state.gate_history1, 'pink', 'circuit1')

    if st.button('Z Gate (Qubit 1)'):
        st.session_state.state_vector1 = apply_z_gate(st.session_state.state_vector1)
        st.session_state.gate_history1.append('Z')
        #update_system(bloch_vector1, st.session_state.state_vector1, st.session_state.gate_history1, 'pink', 'circuit1')

    if st.button('H Gate (Qubit 1)'):
        st.session_state.state_vector1 = apply_h_gate(st.session_state.state_vector1)
        st.session_state.gate_history1.append('H')

    st.write("Apply a qubit channel:")

    px1_bar = st.slider("Rotation X angle Q1", 0.0, 1.0, 0.1)
    if st.button('Apply Rx Channel Q1'):
        st.session_state.state_vector1 = apply_rotation_x_gate(st.session_state.state_vector1, px1_bar)
        st.session_state.gate_history1.append('Rx')

    py1_bar = st.slider("Rotation Y angle Q1", 0.0, 1.0, 0.1)
    if st.button('Apply Ry Channel Q1'):
        st.session_state.state_vector1 = apply_rotation_y_gate(st.session_state.state_vector1, py1_bar)
        st.session_state.gate_history1.append('Ry')

    pz1_bar = st.slider("Rotation Z angle Q1", 0.0, 1.0, 0.1)
    if st.button('Apply Rz Channel Q1'):
        st.session_state.state_vector1 = apply_rotation_z_gate(st.session_state.state_vector1, pz1_bar)
        st.session_state.gate_history1.append('Rz')

    p1_damping = st.slider("Amplitude Damping Probability Q1", 0.0, 1.0, 0.1)
    if st.button('Apply Amplitude Damping Channel Q1'):
        st.session_state.state_vector1 = apply_rotation_z_gate(st.session_state.state_vector1, p1_damping)
        st.session_state.gate_history1.append(f'Amplitude Damping (p={p1_damping:.2f})')

    p1_dephasing = st.slider("Dephasing Probability Q1", 0.0, 1.0, 0.1)
    if st.button('Apply Dephasing Channel Q1'):
        st.session_state.state_vector1 = apply_rotation_z_gate(st.session_state.state_vector1, p1_dephasing)
        st.session_state.gate_history1.append(f'Dephasing (p={p1_dephasing:.2f})')
    update_system(bloch_vector1, st.session_state.state_vector1, st.session_state.gate_history1, 'pink', 'circuit1')



with col3:
    st.header("Qubit 2")
    if 'initial_state_vector2' not in st.session_state:
        st.session_state.initial_state_vector2 = generate_random_state()

    if 'state_vector2' not in st.session_state:
        st.session_state.state_vector2 = st.session_state.initial_state_vector2

    initial_state_vector2 = st.session_state.initial_state_vector2

        # Apply gates to the initial state to get the final state
    final_state_vector2 = apply_h_gate(apply_z_gate(apply_y_gate(apply_x_gate(initial_state_vector2))))
    final_bloch_vector2 = state_to_bloch_vector(final_state_vector2)
    bloch_vector2 = state_to_bloch_vector(st.session_state.state_vector2)
    plot_bloch_sphere(bloch_vector2, final_bloch_vector2, 'ice', "Bloch Sphere 2")
    display_state_and_rho(st.session_state.state_vector2)

    st.write("Apply a gate to Qubit 2:")
    if st.button('X Gate (Qubit 2)'):
        st.session_state.state_vector2 = apply_x_gate(st.session_state.state_vector2)
        st.session_state.gate_history2.append('X')
        #update_system(bloch_vector2, st.session_state.state_vector2, st.session_state.gate_history2, 'lightblue','circuit2')
    if st.button('Y Gate (Qubit 2)'):
        st.session_state.state_vector2 = apply_y_gate(st.session_state.state_vector2)
        st.session_state.gate_history2.append('Y')
        #update_system(bloch_vector2, st.session_state.state_vector2, st.session_state.gate_history2, 'lightblue','circuit2')

    if st.button('Z Gate (Qubit 2)'):
        st.session_state.state_vector2 = apply_z_gate(st.session_state.state_vector2)
        st.session_state.gate_history2.append('Z')
        #update_system(bloch_vector2, st.session_state.state_vector2, st.session_state.gate_history2, 'lightblue','circuit2')

    if st.button('H Gate (Qubit 2)'):
        st.session_state.state_vector2 = apply_h_gate(st.session_state.state_vector2)
        st.session_state.gate_history2.append('H')

    st.write("Apply a qubit channelss:")

    px2_bar = st.slider("Rotation X Q2 angle", 0.0, 1.0, 0.1)
    if st.button('Apply Rx Channel Q2'):
        st.session_state.state_vector2 = apply_rotation_x_gate(st.session_state.state_vector1,px2_bar)
        st.session_state.gate_history2.append('Rx')

    py2_bar = st.slider("Rotation Y Q2 angle", 0.0, 1.0, 0.1)
    if st.button('Apply Ry Channel Q2'):
        st.session_state.state_vector2 = apply_rotation_y_gate(st.session_state.state_vector1,py2_bar)
        st.session_state.gate_history2.append('Ry')

    pz2_bar = st.slider("Rotation Z Q2 angle", 0.0, 1.0, 0.1)
    if st.button('Apply Rz Channel Q2'):
        st.session_state.state_vector2 = apply_rotation_z_gate(st.session_state.state_vector1,pz2_bar)
        st.session_state.gate_history2.append('Rz')

    p2_damping = st.slider("Amplitude Damping Probability Q2", 0.0, 1.0, 0.1)
    if st.button('Apply Amplitude Damping Channel Q2'):
        st.session_state.state_vector2 = apply_rotation_z_gate(st.session_state.state_vector1,p2_damping)
        st.session_state.gate_history2.append(f'Amplitude Damping (p={p2_damping:.2f})')


    p2_dephasing = st.slider("Dephasing Probability Q2", 0.0, 1.0, 0.1)
    if st.button('Apply Dephasing Channel Q2'):
        st.session_state.state_vector2 = apply_rotation_z_gate(st.session_state.state_vector1,p2_dephasing)
        st.session_state.gate_history2.append(f'Dephasing (p={p2_dephasing:.2f})')
    update_system(bloch_vector2, st.session_state.state_vector2, st.session_state.gate_history2, 'lightblue','circuit2')

#with st.container():
 #   st.plotly_chart(circuit_fig)

st.write("Your task is to transform the Bloch vector to a final target state!")

