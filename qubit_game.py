import plotly.graph_objects as go
import numpy as np
import streamlit as st

st.title('BlochOut')

# Initialize session state for gate history if it doesn't exist
if 'gate_history' not in st.session_state:
    st.session_state.gate_history = []

# Function to convert the 2D quantum state to a 3D Bloch vector
def state_to_bloch_vector(state):
    alpha, beta = state
    x = 2 * np.real(alpha * np.conj(beta))
    y = 2 * np.imag(alpha * np.conj(beta))
    z = np.abs(alpha)**2 - np.abs(beta)**2
    return np.array([x, y, z])

# Function to generate a random quantum state
def generate_random_state():
    r = np.random.uniform(0, 1)
    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2 * np.pi)
    alpha = r * np.cos(theta / 2)
    beta = r * np.sin(theta / 2) * np.exp(1j * phi)
    return [alpha, beta]

def compute_pauli_expansion(state_vector):
    alpha, beta = state_vector
    expectation_x = np.real(alpha * np.conj(beta))
    expectation_y = np.imag(alpha * np.conj(beta))
    expectation_z = np.abs(alpha) ** 2 - np.abs(beta) ** 2
    rho = 0.5 * (np.eye(2) + expectation_x * np.array([[0, 1], [1, 0]]) +
                 expectation_y * np.array([[0, -1j], [1j, 0]]) +
                 expectation_z * np.array([[1, 0], [0, -1]]))
    return rho, expectation_x, expectation_y, expectation_z

# Functions for qubit channels
def amplitude_damping_channel(rho, p):
    E0 = np.array([[1, 0], [0, np.sqrt(1 - p)]])
    E1 = np.array([[0, np.sqrt(p)], [0, 0]])
    return E0 @ rho @ E0.conj().T + E1 @ rho @ E1.conj().T

def dephasing_channel(rho, p):
    E0 = np.sqrt(1 - p) * np.eye(2)
    E1 = np.sqrt(p) * np.array([[1, 0], [0, -1]])
    return E0 @ rho @ E0.conj().T + E1 @ rho @ E1.conj().T

# Function to extract Bloch vector from density matrix
def density_matrix_to_bloch_vector(rho):
    x = 2 * np.real(rho[0, 1])
    y = 2 * np.imag(rho[0, 1])
    z = np.real(rho[0, 0] - rho[1, 1])
    return np.array([x, y, z])

# Gate functions
def apply_x_gate(state_vector):
    x_matrix = np.array([[0, 1], [1, 0]])
    return np.dot(x_matrix, state_vector)

def apply_h_gate(state_vector):
    h_matrix = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
    return np.dot(h_matrix, state_vector)

def apply_z_gate(state_vector):
    z_matrix = np.array([[1, 0], [0, -1]])
    return np.dot(z_matrix, state_vector)

def apply_y_gate(state_vector):
    y_matrix = np.array([[0, -1j], [1j, 0]])
    return np.dot(y_matrix, state_vector)

# Initialize the figure object (only once)
fig = go.Figure()

# Function to update the Bloch sphere plot with a new state vector
def plot_bloch_sphere(bloch_vector, final_bloch):
    fig.data = []  # Clear any previous traces

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    fig.add_surface(x=x, y=y, z=z, colorscale='pinkyl', opacity=0.6)

    # Plot for the first bloch vector
    fig.add_trace(go.Scatter3d(
        x=[bloch_vector[0]], y=[bloch_vector[1]], z=[bloch_vector[2]],
        mode='markers+text',
        marker=dict(size=5, color='black'),
        name='State Vector',
        text=['State'],
        textposition='top center'
    ))

    fig.add_trace(go.Scatter3d(
        x=[0, bloch_vector[0]], y=[0, bloch_vector[1]], z=[0, bloch_vector[2]],
        mode='lines+text',
        line=dict(color='black', width=4),
        name="Arrow to State"
    ))

    # Plot for the final_bloch vector
    fig.add_trace(go.Scatter3d(
        x=[final_bloch[0]], y=[final_bloch[1]], z=[final_bloch[2]],
        mode='markers+text',
        marker=dict(size=5, color='white'),
        name='Final State Vector',
        text=['Final State'],
        textposition='top center'
    ))

    fig.add_trace(go.Scatter3d(
        x=[0, final_bloch[0]], y=[0, final_bloch[1]], z=[0, final_bloch[2]],
        mode='lines+text',
        line=dict(color='white', width=4),
        name="Arrow to Final State"
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=4, range=[-1, 1]),
            yaxis=dict(nticks=4, range=[-1, 1]),
            zaxis=dict(nticks=4, range=[-1, 1]),
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        title="Bloch Sphere with State Vectors"
    )

# Initialize the wire layout
circuit_fig = go.Figure()

# Function to draw the quantum circuit
def draw_circuit():
    circuit_fig.data = []  # Clear previous circuit

    circuit_fig.add_trace(go.Scatter(
        x=[0, 5],
        y=[0, 0],
        mode="lines",
        line=dict(width=3, color="black"),
        name="Wire"
    ))

    for i, gate in enumerate(st.session_state.gate_history):
        circuit_fig.add_trace(go.Scatter(
            x=[i + 0.5], y=[0],
            mode="markers+text",
            marker=dict(symbol="square", size=40, color='pink'),
            text=[gate],
            textposition='middle center',
            name=f"Gate {i + 1}"
        ))

    circuit_fig.update_layout(
        xaxis=dict(range=[0, 5], zeroline=False, showticklabels=False),
        yaxis=dict(range=[-1, 1], zeroline=False, showticklabels=False),
        margin=dict(l=0, r=0, b=0, t=0),
        height=200,
        showlegend=False
    )

# Generate a random initial state vector
if 'initial_state_vector' not in st.session_state:
    st.session_state.initial_state_vector = generate_random_state()

# Store the current state from the initial state (or any updates)
if 'state_vector' not in st.session_state:
    st.session_state.state_vector = st.session_state.initial_state_vector

initial_state_vector = st.session_state.initial_state_vector

# Apply gates to the initial state to get the final state
final_state_vector = apply_h_gate(apply_z_gate(apply_y_gate(apply_x_gate(initial_state_vector))))

# Random order of gates to reach a final state
#final_state_vector = apply_h_gate(apply_z_gate(apply_y_gate(apply_x_gate(st.session_state.state_vector))))

bloch_vector = state_to_bloch_vector(st.session_state.state_vector)
final_bloch_vector = state_to_bloch_vector(final_state_vector)

plot_bloch_sphere(bloch_vector,final_bloch_vector)

def display_state_and_rho():
    alpha, beta = st.session_state.state_vector
    state_string = f"$|\psi> = ({alpha:.2f})|0> + ({beta:.2f})|1>$"
    rho, expectation_x, expectation_y, expectation_z = compute_pauli_expansion(st.session_state.state_vector)
    rho_string = f"Density Matrix in Pauli Basis: \nρ = 0.5 (I + {expectation_x:.2f}σx + {expectation_y:.2f}σy + {expectation_z:.2f}σz)"
    st.write(state_string)
    st.write(rho_string)

with st.container():
    st.plotly_chart(fig)
    display_state_and_rho()

st.write("Choose a gate to apply to the state vector:")

# Gate buttons in a horizontal layout
col1, col2, col3, col4 = st.columns(4)

def update_state(new_state_vector):
    #global final_bloch_vector
    st.session_state.state_vector = new_state_vector
    bloch_vector = state_to_bloch_vector(st.session_state.state_vector)
    plot_bloch_sphere(bloch_vector, final_bloch_vector)
    draw_circuit()
    #display_state_and_rho()

with col1:
    if st.button('Apply Pauli-X Gate'):
        update_state(apply_x_gate(st.session_state.state_vector))
        st.session_state.gate_history.append('X')
        if len(st.session_state.gate_history) > 5:
            st.session_state.gate_history.pop(0)

with col2:
    if st.button('Apply Hadamard Gate'):
        update_state(apply_h_gate(st.session_state.state_vector))
        st.session_state.gate_history.append('H')
        if len(st.session_state.gate_history) > 5:
            st.session_state.gate_history.pop(0)

with col3:
    if st.button('Apply Pauli-Z Gate'):
        update_state(apply_z_gate(st.session_state.state_vector))
        st.session_state.gate_history.append('Z')
        if len(st.session_state.gate_history) > 5:
            st.session_state.gate_history.pop(0)

with col4:
    if st.button('Apply Pauli-Y Gate'):
        update_state(apply_y_gate(st.session_state.state_vector))
        st.session_state.gate_history.append('Y')
        if len(st.session_state.gate_history) > 5:
            st.session_state.gate_history.pop(0)

# Channel buttons
st.write("Apply a qubit channel:")

p_damping = st.slider("Amplitude Damping Probability", 0.0, 1.0, 0.1)
if st.button('Apply Amplitude Damping Channel'):
    rho = amplitude_damping_channel(rho, p_damping)
    st.session_state.gate_history.append(f'Amplitude Damping (p={p_damping:.2f})')
    if len(st.session_state.gate_history) > 5:
        st.session_state.gate_history.pop(0)
    draw_circuit()
    bloch_vector = density_matrix_to_bloch_vector(rho)
    plot_bloch_sphere(bloch_vector, final_bloch_vector)
    #display_state_and_rho()

p_dephasing = st.slider("Dephasing Probability", 0.0, 1.0, 0.1)
if st.button('Apply Dephasing Channel'):
    rho = dephasing_channel(rho, p_dephasing)
    st.session_state.gate_history.append(f'Dephasing (p={p_dephasing:.2f})')
    if len(st.session_state.gate_history) > 5:
        st.session_state.gate_history.pop(0)
    draw_circuit()
    bloch_vector = density_matrix_to_bloch_vector(rho)
    plot_bloch_sphere(bloch_vector, final_bloch_vector)
    #display_state_and_rho()

with st.container():
    st.plotly_chart(circuit_fig)


st.write("Your task is to transform the Bloch vector to a final target state!")

