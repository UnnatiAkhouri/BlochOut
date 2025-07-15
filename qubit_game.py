import plotly.graph_objects as go
import numpy as np
import streamlit as st
from qutip import *
import random
import matplotlib.pyplot as plt
import time
import base64
import os

#Setups for the game visuals and backgrounds
st.set_page_config(layout="wide")

#Code for adding a slider for channels like phase damping and amplitude damping
st.markdown("""
    <style>
    .stSlider [data-baseweb=slider] {
        width: 65%;
    }
    /* Target multiple possible selectors for the slider value */
    [data-testid="stSliderFormattedValue"],
    .stSlider p,
    .stSlider div[data-baseweb="typography"],
    .stSlider [role="slider"] + div {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

#font size and button size
def set_custom_style():
    st.markdown("""
        <style>
        /* Increase base font size for all text */
        .stApp {
            font-size: 30px;
        }
        /* Hide slider formatted value */
        [data-testid="stSliderFormattedValue"] {
            display: none !important;
        }
        
        /* Or alternatively, style it properly */
        /*
        [data-testid="stSliderFormattedValue"] {
            font-size: 1.5em !important;
            width: 5em !important;
            white-space: nowrap !important;
            overflow: hidden !important;
        }

        /* Increase font size for headers */
        h1 {
            font-size: 4.5em !important;
        }

        h2 {
            font-size: 2.5em !important;
        }

        h3 {
            font-size: 2.5em !important;
        }

        /* Increase font size for paragraph text */
        p {
            font-size: 2.0em !important;
        }

        /* Increase font size for markdown text */
        .markdown-text-container {
            font-size: 1.2em !important;
        }

        /* Make buttons larger */
        .stButton>button {
            font-size: 1.5em !important;
            padding: 0.2em 1.8em !important;
            height: auto;
        }

        /* Increase font size for input widgets */
        .stTextInput>div>div>input {
            font-size: 1.2em !important;
        }

        .stSelectbox>div>div>select {
            font-size: 1.2em !important;
        }

        /* Increase size of radio buttons and checkboxes */
        .stRadio>div {
            font-size: 1.2em !important;
            padding: 10px !important;
        }

        .stCheckbox>div {
            font-size: 1.2em !important;
            padding: 10px !important;
        }

        /* Increase slider size and text */
        .stSlider>div>div {
            font-size: 0.5em !important;
        }

        /* Increase size of number inputs */
        .stNumberInput>div>div>input {
            font-size: 1.2em !important;
            padding: 0.5em !important;
        }

        /* Increase sidebar font size */
        .sidebar .sidebar-content {
            font-size: 1.2em !important;
        }

        /* Increase metrics (st.metric) font size */
        .stMetric>div {
            font-size: 1.2em !important;
        }

        /* Increase dataframe/table font size */
        .dataframe {
            font-size: 1.2em !important;
        }

        /* Increase size of file uploader */
        .stFileUploader>div>div {
            font-size: 1.2em !important;
            padding: 1em !important;
        }

        /* Style for large buttons specifically */
        .big-button {
            font-size: 2.5em !important;
            padding: 1em 2em !important;
            margin: 0.5em 0 !important;
            width: 100% !important;
        }
        </style>
    """, unsafe_allow_html=True)


def set_background(image_path):
    """Set page background to specified image"""
    with open(image_path, "rb") as img_file:
        encoded_img = base64.b64encode(img_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-attachment: scroll !important;
            background-image: url("data:image/png;base64,{encoded_img}");
            background-size: auto;
            background-repeat: no-repeat;
            background-position: center;
            min-height: 150vh;
        }}

        /* Override any fixed positioning that might affect scrolling */
        .main {{
            background-color: transparent !important;
        }}

        /* Make sure content containers are transparent */
        .block-container, [data-testid="stVerticalBlock"] {{
            background-color: transparent !important;
            padding-left: 0px !important;
            padding-right: 0px !important;
            max-width: 80% !important;
        }}

        /* Make sure plots have transparent backgrounds */
        .js-plotly-plot .plotly .main-svg {{
            background-color: transparent !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

set_background("backgrounds/back.PNG")
#Title of the game
#st.title('BlochOut - Dual Qubits')

# Initialize session state for gate history if it doesn't exist
if 'gate_history1' not in st.session_state:
    st.session_state.gate_history1 = []
if 'gate_history2' not in st.session_state:
    st.session_state.gate_history2 = []


# Function to generate a random two-qubit state



# Hardcoded Bell States
def bell_states():
    phi_plus = (1 / np.sqrt(2)) * (tensor(basis(2, 0), basis(2, 0)) + tensor(basis(2, 1), basis(2, 1)))
    phi_minus = (1 / np.sqrt(2)) * (tensor(basis(2, 0), basis(2, 0)) - tensor(basis(2, 1), basis(2, 1)))
    psi_plus = (1 / np.sqrt(2)) * (tensor(basis(2, 0), basis(2, 1)) + tensor(basis(2, 1), basis(2, 0)))
    psi_minus = (1 / np.sqrt(2)) * (tensor(basis(2, 0), basis(2, 1)) - tensor(basis(2, 1), basis(2, 0)))
    return phi_plus, phi_minus, psi_plus, psi_minus


# Hardcoded W-state
def w_state():
    return (1 / np.sqrt(3)) * (
                tensor(basis(2, 0), basis(2, 0)) + tensor(basis(2, 0), basis(2, 1)) + tensor(basis(2, 1), basis(2, 0)))


# Hardcoded GHZ state
def ghz_state():
    return (1 / np.sqrt(2)) * (tensor(basis(2, 0), basis(2, 0)) + tensor(basis(2, 1), basis(2, 1)))


def partial_trace(state, qubit_num):
    """Compute the partial trace over the specified qubit."""
    # Ensure the state is a Qobj
    if not isinstance(state, Qobj):
        state = Qobj(state)  # Convert to Qobj if it's not already

    # Trace out qubit 0 or 1
    if qubit_num == 0:
        return state.ptrace([1])  # Trace out qubit 0, keeping qubit 1
    elif qubit_num == 1:
        return state.ptrace([0])  # Trace out qubit 1, keeping qubit 0
    else:
        raise ValueError("Invalid qubit number, choose either 0 or 1.")


# Example usage
#if __name__ == '__main__':
    # Generate a random two-qubit state

#random_state = Qobj(random_state)
#random_state = random_state *random_state.dag()

#print("Random two-qubit state:")
#print(random_state)

    # Print Bell states
bell_phi_plus, bell_phi_minus, bell_psi_plus, bell_psi_minus = bell_states()

bell_phi_plus=bell_phi_plus.full()
bell_phi_plus = np.outer(bell_phi_plus, np.conj(bell_phi_plus))

bell_phi_minus= bell_phi_minus.full()
bell_phi_minus = np.outer(bell_phi_minus, np.conj(bell_phi_minus))

bell_psi_plus= bell_psi_plus.full()
bell_psi_plus = np.outer(bell_psi_plus, np.conj(bell_psi_plus))

bell_psi_minus= bell_psi_minus.full()
bell_psi_minus = np.outer(bell_psi_minus, np.conj(bell_psi_minus))

w = w_state().full()
#w = Qobj(w)
w = np.outer(w, np.conj(w))

def create_deutsch_initial_state():
    """Create initial state for Deutsch: |01⟩"""
    return np.array([0, 1, 0, 0])

def create_deutsch_density_matrix():
    """Create Deutsch algorithm initial density matrix"""
    psi = create_deutsch_initial_state()
    return np.outer(psi, np.conj(psi))

def apply_hadamard_to_first_qubit(rho_2qubit):
    """Apply Hadamard gate to first qubit of two-qubit system"""
    h_gate = hadamard()
    h_2qubit = np.kron(h_gate, np.eye(2))
    return h_2qubit @ rho_2qubit @ h_2qubit.conj().T

def oracle_constant(rho_2qubit):
    """Oracle that always returns 0 (does nothing)"""
    return rho_2qubit

def oracle_balanced(rho_2qubit):
    """Oracle for balanced function (CNOT gate)"""
    cnot = cnot_gate()
    return cnot @ rho_2qubit @ cnot.conj().T

def create_bell_state_density():
    """Create Bell state (|00⟩ + |11⟩)/√2"""
    psi = (1/np.sqrt(2)) * np.array([1, 0, 0, 1])
    return np.outer(psi, np.conj(psi))

#print("\nW-State:", w)

    # Print GHZ-state
ghz = ghz_state().full()
#ghz = Qobj(ghz)
ghz = np.outer(ghz, np.conj(ghz))
#print("\nGHZ-State:", ghz)

def extract_rho_1(rho_2qubit):
    return np.array([[rho_2qubit[0, 0] + rho_2qubit[1, 1], rho_2qubit[0, 2] + rho_2qubit[1, 3]],
                  [rho_2qubit[2, 0] + rho_2qubit[3, 1], rho_2qubit[2, 2] + rho_2qubit[3, 3]]])

# Reduced density matrix for the second qubit (tracing out the first qubit)
def extract_rho_2(rho_2qubit):
    return np.array([[rho_2qubit[2, 2] + rho_2qubit[0, 0], rho_2qubit[2, 3] + rho_2qubit[0, 1]],
                  [rho_2qubit[3, 2] + rho_2qubit[1, 0], rho_2qubit[3, 3] + rho_2qubit[1, 1]]])


#rho_2qubit_initial = random_state
#print(ghz.ptrace([1]))
    # Compute partial trace to get Bloch vectors
#single_qubit_state_0 = rho_1(random_state)  # Trace out qubit 1, keep qubit 0
#print(single_qubit_state_0)
#single_qubit_state_1 = rho_2(random_state)  # Trace out qubit 1, keep qubit 0
#print(single_qubit_state_1)

    #print("\nSingle Qubit Bloch Vector (Traced out qubit 0):")
    #print(single_qubit_state_0)
    #print("\nSingle Qubit Bloch Vector (Traced out qubit 1):")
    #print(single_qubit_state_1)

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


sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

# Identity matrix
I = np.eye(2)


def rho_1(rho_2qubit):
    """Extract the state of the FIRST qubit by tracing out the second qubit"""
    return np.array([[rho_2qubit[0, 0] + rho_2qubit[1, 1], rho_2qubit[0, 2] + rho_2qubit[1, 3]],
                     [rho_2qubit[2, 0] + rho_2qubit[3, 1], rho_2qubit[2, 2] + rho_2qubit[3, 3]]])

def rho_2(rho_2qubit):
    """Extract the state of the SECOND qubit by tracing out the first qubit"""
    return np.array([[rho_2qubit[2, 2] + rho_2qubit[0, 0], rho_2qubit[2, 3] + rho_2qubit[0, 1]],
                     [rho_2qubit[3, 2] + rho_2qubit[1, 0], rho_2qubit[3, 3] + rho_2qubit[1, 1]]])

def bloch_vector(rho):
    """Compute the Bloch vector from a given density matrix rho."""
    # Compute the expectation values of the Pauli matrices
    r_x = np.real(np.trace(np.dot(rho, sigma_x)))
    r_y = np.real(np.trace(np.dot(rho, sigma_y)))
    r_z = np.real(np.trace(np.dot(rho, sigma_z)))

    # Return the Bloch vector components
    return np.array([r_x, r_y, r_z])

# Function to generate a random quantum state
def generate_random_state():
    r = np.random.uniform(0, 1)
    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2 * np.pi)
    alpha = r * np.cos(theta / 2)
    beta = r * np.sin(theta / 2) * np.exp(1j * phi)
    return [alpha, beta]

def two_qubit_swap_state(p1,p2,theta,corr):
    return np.array([[(1-p1)*(1-p2), 0,corr,corr], [0,p2*np.cos(theta)** +p1*np.sin(theta)** - p1*p2,(p2-p1)*np.sin(theta)*np.cos(theta),0],
              [corr,p1*np.cos(theta)** +p2*np.sin(theta)** - p1*p2,(p2-p1)*np.sin(theta)*np.cos(theta),corr],[corr, 0,corr,p1*p2]])

# Gate functions
import numpy as np
def cz_gate():
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, -1]])

def cr_gate(theta):
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, np.exp(1j)]])

def pswap_gate(theta):
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta), 1j * np.sin(theta), 0],
        [0, 1j * np.sin(theta), np.cos(theta), 0],
        [0, 0, 0, 1]
    ])

def apply_x_gate(rho):
    """Apply X gate to the given density matrix rho."""
    x_gate = np.array([[0, 1], [1, 0]])
    return np.dot(x_gate, np.dot(rho, x_gate.conj().T))

def apply_y_gate(rho):
    """Apply Y gate to the given density matrix rho."""
    y_gate = np.array([[0, -1j], [1j, 0]])
    return np.dot(y_gate, np.dot(rho, y_gate.conj().T))

def apply_z_gate(rho):
    """Apply Z gate to the given density matrix rho."""
    z_gate = np.array([[1, 0], [0, -1]])
    return np.dot(z_gate, np.dot(rho, z_gate.conj().T))

def apply_h_gate(rho):
    """Apply H gate (Hadamard) to the given density matrix rho."""
    h_gate = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
    return np.dot(h_gate, np.dot(rho, h_gate.conj().T))

def apply_s_gate(rho):
    s_gate= np.array([[1, 0],[0, 1j]])
    return np.dot(s_gate, np.dot(rho, s_gate.conj().T))

def apply_rotation_x_gate(rho, p):
    """Apply rotation gate R_x(p) to the density matrix rho."""
    rx_matrix = np.array([[np.cos(p/2), -1j * np.sin(p/2)],
                          [-1j * np.sin(p/2), np.cos(p/2)]])
    return np.dot(rx_matrix, np.dot(rho, rx_matrix.conj().T))

def apply_rotation_y_gate(rho, p):
    """Apply rotation gate R_y(p) to the density matrix rho."""
    ry_matrix = np.array([[np.cos(p/2), -np.sin(p/2)],
                          [np.sin(p/2), np.cos(p/2)]])
    return np.dot(ry_matrix, np.dot(rho, ry_matrix.conj().T))

def apply_rotation_z_gate(rho, p):
    """Apply rotation gate R_z(p) to the density matrix rho."""
    rz_matrix = np.array([[np.exp(-1j * p / 2), 0],
                          [0, np.exp(1j * p / 2)]])
    return np.dot(rz_matrix, np.dot(rho, rz_matrix.conj().T))

def amplitude_damping_channel(rho, p):
    """Apply amplitude damping channel to the density matrix rho."""
    E0 = np.array([[1, 0], [0, np.sqrt(1 - p)]])
    E1 = np.array([[0, np.sqrt(p)], [0, 0]])
    new_rho = np.dot(E0, np.dot(rho, E0.conj().T)) + np.dot(E1, np.dot(rho, E1.conj().T))
    return new_rho

def dephasing_channel(rho, p):
    """Apply dephasing channel to the density matrix rho."""
    E0 = np.sqrt(1 - p) * np.eye(2)
    E1 = np.sqrt(p) * np.array([[1, 0], [0, -1]])
    new_rho = np.dot(E0, np.dot(rho, E0.conj().T)) + np.dot(E1, np.dot(rho, E1.conj().T))
    return new_rho

def hadamard():
    return np.array([[1, 1], [1, -1]]) / np.sqrt(2)

def pauli_x():
    return np.array([[0, 1], [1, 0]])

def pauli_y():
    return np.array([[0, -1j], [1j, 0]])

def pauli_z():
    return np.array([[1, 0], [0, -1]])

def s_gate():
    return np.array([[1, 0],
                     [0, 1j]])

def t_gate():
    return np.array([[1, 0],
                     [0, np.exp(1j * np.pi / 4)]])

def r_x_rotation(p):
    return np.array([[np.cos(p / 2), -1j * np.sin(p / 2)],
                          [-1j * np.sin(p / 2), np.cos(p / 2)]])

def r_y_rotation(p):
    return np.array([[np.cos(p/2), -np.sin(p/2)],
                          [np.sin(p/2), np.cos(p/2)]])

def r_z_rotation(p):
    return np.array([[np.exp(-1j * p / 2), 0],
                          [0, np.exp(1j * p / 2)]])
#Two qubit gates
def cnot_gate():
    # 2-qubit CNOT
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]])



#if 'initial_state_rho1' not in st.session_state:
#    st.session_state.initial_state_rho1 = single_qubit_state_0

# Initialize the first and second state vectors
#if 'state_rho1' not in st.session_state:
    #st.session_state.state_vector1 = generate_random_state()

#    st.session_state.state_rho1 = st.session_state.initial_state_rho1

 #   initial_state_rho1 = st.session_state.initial_state_rho1

    # Apply gates to the initial state to get the final state
#   final_state_rho1 = apply_z_gate(apply_z_gate(apply_y_gate(apply_x_gate(initial_state_rho1))))
#    final_bloch_rho1 = bloch_vector(final_state_rho1)

#if 'initial_state_rho2' not in st.session_state:
#    st.session_state.initial_state_rho2 = single_qubit_state_0

#if 'state_rho2' not in st.session_state:
#    st.session_state.state_rho2 = st.session_state.initial_state_rho2

#    initial_state_rho2 = st.session_state.initial_state_rho2

    # Apply gates to the initial state to get the final state
#    final_state_rho2 = apply_h_gate(apply_z_gate(apply_y_gate(apply_x_gate(initial_state_rho2))))
#    final_bloch_rho2 = bloch_vector(final_state_rho2)

# In the marker dict, add:
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


    st.markdown(
        """
        <style>
        button[kind="primary"] {
            font-family: 'Comic Sans MS', cursive, sans-serif;
            font-size: 26px;
            color: #4B0082;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

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
        ),autosize=False, width=650, height=650,
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the entire figure
        margin=dict(l=0, r=0, b=0, t=0),
        #title=title
    )
    st.plotly_chart(fig)


# Function to update Bloch sphere and circuit
def update_system(bloch_vector, state__rho, gate_history, circuit_color, circuit_key):
    #bloch_vector[:] = state_to_bloch_vector(state_vector)
    #bloch_vector
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
            marker=dict(symbol="square", size=70, color=color),
            text=[gate],
            textposition='middle center',
            textfont=dict(size=25, color="black"),
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


def draw_two_qubit_circuit(gate_history1, gate_history2):
    """
    Draw a two-qubit circuit showing the last 5 gates applied to each qubit.

    Parameters:
    gate_history1 (list): List of gates applied to qubit 1
    gate_history2 (list): List of gates applied to qubit 2
    """
    max_gates = 5
    # Get the last 5 gates or fewer if not enough gates
    truncated_history1 = gate_history1[-max_gates:] if len(gate_history1) > 0 else []
    truncated_history2 = gate_history2[-max_gates:] if len(gate_history2) > 0 else []

    # Calculate the maximum number of gates to show
    max_cols = max(len(truncated_history1), len(truncated_history2), 1)

    # Create the figure
    circuit_fig = go.Figure()

    # Add the qubit wires
    circuit_fig.add_trace(go.Scatter(
        x=[0, max_cols + 1],
        y=[1, 1],
        mode="lines",
        line=dict(width=3, color="black"),
        name="Qubit 1 Wire"
    ))

    circuit_fig.add_trace(go.Scatter(
        x=[0, max_cols + 1],
        y=[0, 0],
        mode="lines",
        line=dict(width=3, color="black"),
        name="Qubit 2 Wire"
    ))

    # Add labels for the qubits
    circuit_fig.add_annotation(
        x=0, y=1,
        text="Qubit 1",
        showarrow=False,
        font=dict(size=14, color="black")
    )

    circuit_fig.add_annotation(
        x=0, y=0,
        text="Qubit 2",
        showarrow=False,
        font=dict(size=14, color="black")
    )

    # Add gates for qubit 1
    for i, gate in enumerate(truncated_history1):
        circuit_fig.add_trace(go.Scatter(
            x=[i + 1], y=[1],
            mode="markers+text",
            marker=dict(symbol="square", size=70, color="pink"),
            text=[gate],
            textposition='middle center',
            textfont=dict(size=25, color="black"),
            name=f"Qubit 1 Gate {i + 1}"
        ))

    # Add gates for qubit 2
    for i, gate in enumerate(truncated_history2):
        circuit_fig.add_trace(go.Scatter(
            x=[i + 1], y=[0],
            mode="markers+text",
            marker=dict(symbol="square", size=70, color="skyblue"),
            text=[gate],
            textposition='middle center',
            textfont=dict(size=25, color="black"),
            name=f"Qubit 2 Gate {i + 1}"
        ))

    # Update layout
    circuit_fig.update_layout(
        xaxis=dict(range=[-0.5, max_cols + 1.5], zeroline=False, showticklabels=False),
        yaxis=dict(range=[-0.5, 1.5], zeroline=False, showticklabels=False),
        margin=dict(l=30, r=30, b=30, t=30),
        height=200,
        width=500,
        showlegend=False,
        title="Two-Qubit Circuit History (Last 5 Gates)"
    )

    # Display the circuit - use a unique key based on the lengths of both histories
    st.plotly_chart(circuit_fig, key=f"two_qubit_circuit_{len(gate_history1)}_{len(gate_history2)}")

# Function to display state and rho
def display_state_and_rho(state_vector):
    #alpha, beta = state_vector
    #state_string = f"$|\psi> = ({alpha:.2f})|0> + ({beta:.2f})|1>$"
    rho, expectation_x, expectation_y, expectation_z = compute_pauli_expansion(state_vector)
    rho_string = f"ρ = 0.5 (I + {expectation_x:.2f}σx + {expectation_y:.2f}σy + {expectation_z:.2f}σz)"
    #st.write(state_string)
    st.write(rho_string)

def display_rho(rho):
    r_x = np.real(np.trace(np.dot(rho, sigma_x)))
    r_y = np.real(np.trace(np.dot(rho, sigma_y)))
    r_z = np.real(np.trace(np.dot(rho, sigma_z)))
    rho_string = f"$\\rho = 0.5 (I + {r_x:.2f}X + {r_y:.2f}Y + {r_z:.2f}Z)$"
    st.markdown(f"<span style='font-size: 35px'>{rho_string}</span>", unsafe_allow_html=True)

def level(score):
    st.subheader("Challenge: Match the Bloch vectors")
    # Add your specific level 1 mechanics
    score  # Your scoring function
    threshold = 45
    if score > threshold:
        return True
    return False

def show_progress():
    total_levels = 5
    progress = (st.session_state.current_level - 1) / total_levels
    st.progress(progress)
    #st.text(f"Level {st.session_state.current_level} of {total_levels}")

if 'score' not in st.session_state:
    st.session_state.score = 0

def update_score(points):
    st.session_state.score += points
    st.sidebar.text(f"Total Score: {st.session_state.score}")

def check_level_requirements():
    if st.session_state.current_level == 2:
        if st.session_state.score < 0:
            st.warning("You need 100 points to unlock this level!")
            return False
    return True

if 'current_level' not in st.session_state:
    st.session_state.current_level = 1

# Function to show loading animation between levels
def level_transition():
    with st.spinner('Loading next level...'):
        time.sleep(2)  # Add a 2-second delay
    st.success('Level Complete!')
    time.sleep(1)  # Show success message for 1 second
    st.session_state.current_level += 1
    if 'state_rho' in st.session_state:
        del st.session_state.state_rho
    if 'state_rho1' in st.session_state:
        del st.session_state.state_rho1
    if 'state_rho2' in st.session_state:
        del st.session_state.state_rho2
    if 'initial_state_rho' in st.session_state:
        del st.session_state.initial_state_rho
    if 'initial_state_rho1' in st.session_state:
        del st.session_state.initial_state_rho1
    if 'initial_state_rho2' in st.session_state:
        del st.session_state.initial_state_rho2
    if 'final_state_rho1' in st.session_state:
        del st.session_state.final_state_rho1

        # Clear gate histories
    if 'gate_history1' in st.session_state:
        del st.session_state.gate_history1
    if 'gate_history2' in st.session_state:
        del st.session_state.gate_history2
    #st.experimental_rerun()

# ============= Quantum Gates =============
def hadamard():
    """Hadamard gate"""
    return np.array([[1, 1], [1, -1]]) / np.sqrt(2)


def pauli_x():
    """Pauli X gate"""
    return np.array([[0, 1], [1, 0]])


def pauli_y():
    """Pauli Y gate"""
    return np.array([[0, -1j], [1j, 0]])


def pauli_z():
    """Pauli Z gate"""
    return np.array([[1, 0], [0, -1]])


def s_gate():
    """S gate"""
    return np.array([[1, 0], [0, 1j]])



def t_gate():
    """T gate"""
    return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])


def r_x_rotation(p):
    """Rotation around X axis"""
    return np.array([
        [np.cos(p / 2), -1j * np.sin(p / 2)],
        [-1j * np.sin(p / 2), np.cos(p / 2)]
    ])

def phi_rotation(p):
    return np.array([
        [np.exp(1j*p), 0],
        [0, np.exp(1j*p)]
    ])


def r_y_rotation(p):
    """Rotation around Y axis"""
    return np.array([
        [np.cos(p / 2), -np.sin(p / 2)],
        [np.sin(p / 2), np.cos(p / 2)]
    ])


def r_z_rotation(p):
    """Rotation around Z axis"""
    return np.array([
        [np.exp(-1j * p / 2), 0],
        [0, np.exp(1j * p / 2)]
    ])


# ============= Two Qubit Gates =============
def cnot_gate():
    """CNOT"""
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ])


def cz_gate():
    """CZ gate"""
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1]
    ])


# ============= Quantum Channels =============
def amplitude_damping_channel(rho, p):
    """Apply amplitude damping channel"""
    E0 = np.array([[1, 0], [0, np.sqrt(1 - p)]])
    E1 = np.array([[0, np.sqrt(p)], [0, 0]])
    return np.dot(E0, np.dot(rho, E0.conj().T)) + np.dot(E1, np.dot(rho, E1.conj().T))


def dephasing_channel(rho, p):
    """Apply dephasing channel"""
    E0 = np.sqrt(1 - p) * np.eye(2)
    E1 = np.sqrt(p) * np.array([[1, 0], [0, -1]])
    return np.dot(E0, np.dot(rho, E0.conj().T)) + np.dot(E1, np.dot(rho, E1.conj().T))


# ============= Bloch Sphere Visualization =============
def plot_bloch_sphere(bloch_vector, final_bloch, color, title):
    """Create and display Bloch sphere visualization with custom markers"""
    fig = go.Figure()

    # Create sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    # Add surface
    fig.add_surface(x=x, y=y, z=z, colorscale=color, opacity=0.6, showscale=False)

    # Function to create door shape
    def create_door(x, y, z, scale=0.1):
        door_x = [x - scale / 2, x + scale / 2, x + scale / 2, x - scale / 2, x - scale / 2]
        door_y = [y - scale / 2, y - scale / 2, y + scale / 2, y + scale / 2, y - scale / 2]
        door_z = [z, z, z, z, z]
        return door_x, door_y, door_z

    # Function to create stick figure
    def create_stick_figure(x, y, z, scale=0.15):
        # Head
        head_x = [x]
        head_y = [y]
        head_z = [z + scale / 2]

        # Body
        body_x = [x, x]
        body_y = [y, y]
        body_z = [z + scale / 2, z - scale / 2]

        # Arms
        arms_x = [x - scale / 2, x + scale / 2]
        arms_y = [y, y]
        arms_z = [z, z]

        # Legs
        legs_x = [x - scale / 3, x + scale / 3]
        legs_y = [y, y]
        legs_z = [z - scale / 2, z - scale / 2]

        return head_x, head_y, head_z, body_x, body_y, body_z, arms_x, arms_y, arms_z, legs_x, legs_y, legs_z

    # Add door for current state

    door_x, door_y, door_z = create_door(final_bloch[0], final_bloch[1], final_bloch[2])
    fig.add_trace(go.Scatter3d(
        x=door_x, y=door_y, z=door_z,
        mode='lines',
        line=dict(color='black', width=4),
        name='State'
    ))

    # Add vector line from origin to door
    fig.add_trace(go.Scatter3d(
        x=[0, bloch_vector[0]], y=[0, bloch_vector[1]], z=[0, bloch_vector[2]],
        mode='lines',
        line=dict(color='black', width=2),
        name=""
    ))

    # Add stick figure for final state
    head_x, head_y, head_z, body_x, body_y, body_z, arms_x, arms_y, arms_z, legs_x, legs_y, legs_z = create_stick_figure(
        bloch_vector[0], bloch_vector[1], bloch_vector[2]
    )

    # Add stick figure components
    fig.add_trace(go.Scatter3d(x=head_x, y=head_y, z=head_z, mode='markers',
                               marker=dict(size=5, color='magenta'), name='Final State'))
    fig.add_trace(go.Scatter3d(x=body_x, y=body_y, z=body_z, mode='lines',
                               line=dict(color='magenta', width=2), showlegend=False))
    fig.add_trace(go.Scatter3d(x=arms_x, y=arms_y, z=arms_z, mode='lines',
                               line=dict(color='magenta', width=2), showlegend=False))
    fig.add_trace(go.Scatter3d(x=legs_x, y=legs_y, z=legs_z, mode='lines',
                               line=dict(color='magenta', width=2), showlegend=False))

    # Add vector line from origin to stick figure
    fig.add_trace(go.Scatter3d(
        x=[0, final_bloch[0]], y=[0, final_bloch[1]], z=[0, final_bloch[2]],
        mode='lines',
        line=dict(color='magenta', width=2),
        showlegend=False
    ))

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=4, range=[-1, 1], showbackground=False, zeroline=False),
            yaxis=dict(nticks=4, range=[-1, 1], showbackground=False, zeroline=False),
            zaxis=dict(nticks=4, range=[-1, 1], showbackground=False, zeroline=False),
            bgcolor='rgba(0,0,0,0)',
        ),
        autosize=False,
        width=550,
        height=550,
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, b=0, t=0),
    )

    st.plotly_chart(fig)

# ============= Game UI Components =============
def draw_circuit(gate_history, color, circuit_key):
    """Draw quantum circuit visualization"""
    max_gates = 5
    truncated_history = gate_history[-max_gates:]
    num_gates = len(truncated_history)

    circuit_fig = go.Figure()

    # Add wire
    circuit_fig.add_trace(go.Scatter(
        x=[0, num_gates + 1],
        y=[0, 0],
        mode="lines",
        line=dict(width=3, color="black"),
        name="Wire"
    ))

    # Add gates
    for i, gate in enumerate(truncated_history):
        circuit_fig.add_trace(go.Scatter(
            x=[i + 1], y=[0],
            mode="markers+text",
            marker=dict(symbol="square", size=70, color=color),
            text=[gate],
            textposition='middle center',
            textfont=dict(size=25, color="black"),
            name=f"Gate {i + 1}"
        ))

    circuit_fig.update_layout(
        xaxis=dict(range=[0, max_gates + 1], zeroline=False, showticklabels=False),
        yaxis=dict(range=[-1, 1], zeroline=False, showticklabels=False),
        margin=dict(l=0, r=0, b=0, t=0),
        height=100,
        width=400,
        showlegend=False
    )

    st.plotly_chart(circuit_fig, key=circuit_key)

def random_two_qubit_state():
    # Randomly generate the complex coefficients alpha, beta, gamma, delta
    alpha = random.uniform(0, 1) + 1j * random.uniform(0, 1)
    beta = random.uniform(0, 1) + 1j * random.uniform(0, 1)
    gamma = random.uniform(0, 1) + 1j * random.uniform(0, 1)
    delta = random.uniform(0, 1) + 1j * random.uniform(0, 1)

    # Normalize the coefficients so that the total probability sums to 1
    norm = np.sqrt(abs(alpha) ** 2 + abs(beta) ** 2 + abs(gamma) ** 2 + abs(delta) ** 2)
    alpha /= norm
    beta /= norm
    gamma /= norm
    delta /= norm

    # Return the normalized quantum state
    return alpha * basis(4, 0) + beta * basis(4, 1) + gamma * basis(4, 2) + delta * basis(4, 3)

def display_density_matrix(state_rho):
    """Display density matrix visualization"""
    st.subheader("Density Matrix Real Parts")

    rho_real = np.real(state_rho)

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
    cax1 = ax1.imshow(rho_real, cmap="BuPu", interpolation="nearest")
    fig.colorbar(cax1, ax=ax1)
    ax1.set_title("Real Part of Density Matrix")

    plt.tight_layout()
    st.pyplot(fig)

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


sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

# Identity matrix
I = np.eye(2)

def one_qubit_state(a1,a2,a3):
    rho_q1 = (1/2)*(np.eye(2) + a1 * sigma_x + a2 * sigma_y + a3* sigma_z)
    rho_q2 = (1/2)*(np.eye(2))
    rho_full = np.kron(rho_q1, rho_q2)
    return rho_full


def bloch_vector(rho):
    """Compute the Bloch vector from a given density matrix rho."""
    # Compute the expectation values of the Pauli matrices
    r_x = np.real(np.trace(np.dot(rho, sigma_x)))
    r_y = np.real(np.trace(np.dot(rho, sigma_y)))
    r_z = np.real(np.trace(np.dot(rho, sigma_z)))

    # Return the Bloch vector components
    return np.array([r_x, r_y, r_z])



# Function to generate a random quantum state
def generate_random_state():
    r = np.random.uniform(0, 1)
    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2 * np.pi)
    alpha = r * np.cos(theta / 2)
    beta = r * np.sin(theta / 2) * np.exp(1j * phi)
    return [alpha, beta]

# ============= Game Logic =============
#def level():
#    """Level 1 game logic"""
    # Add your level 1 winning conditions here
#    pass


def update_score(points):
    """Update game score"""
    if 'score' not in st.session_state:
        st.session_state.score = 0
    st.session_state.score += points


def level_transition():
    """Handle transition between levels"""
    with st.spinner('Loading next level...'):
        time.sleep(2)
    st.success('Level Complete!')
    time.sleep(2)
    st.session_state.current_level += 1
    #st.experimental_rerun()


def add_pdf_download():
    """Add a single PDF download button to your Streamlit app"""

    # Path to your PDF file
    pdf_path = "assets/BlochOut.pdf"

    # Check if PDF exists and create download button
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                label="Download Quantum Guide",
                data=pdf_file.read(),
                file_name="BlochOut.pdf",
                mime="application/pdf"
            )
    else:
        st.warning("⚠️ PDF not found. Please add quantum_guide.pdf to assets/pdfs/")

# ============= Main Game Loop =============
def main():
    global cnot_gate_q1_controls_q2, cr_gate_q1_controls_q2, cnot_gate_q2_controls_q1, cr_gate_q2_controls_q1
    st.title("Escape the Bloch")
    set_custom_style()
    show_progress()
    add_pdf_download()
    if not check_level_requirements():
        return

    # Add a reset button for debugging
    if st.sidebar.button("Reset Game"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    # Initialize current_level if not already done
    if 'current_level' not in st.session_state:
        st.session_state.current_level = 1

    # Level 1 logic
    if st.session_state.current_level == 1:
        st.header("Level 1: Come back to *your-qubit-self!*")

        # Initialize feedback state variables
        if 'feedback_active' not in st.session_state:
            st.session_state.feedback_active = False
            st.session_state.feedback_message = ""
            st.session_state.feedback_type = ""
            st.session_state.gate_effect_message = ""

        # Define your rho_1 and rho_2 functions
        def rho_1(rho_2qubit):
            return np.array([[rho_2qubit[0, 0] + rho_2qubit[1, 1], rho_2qubit[0, 2] + rho_2qubit[1, 3]],
                             [rho_2qubit[2, 0] + rho_2qubit[3, 1], rho_2qubit[2, 2] + rho_2qubit[3, 3]]])

        def rho_2(rho_2qubit):
            return np.array([[rho_2qubit[2, 2] + rho_2qubit[0, 0], rho_2qubit[2, 3] + rho_2qubit[0, 1]],
                             [rho_2qubit[3, 2] + rho_2qubit[1, 0], rho_2qubit[3, 3] + rho_2qubit[1, 1]]])

        # Story and riddle state initialization
        if 'story_index' not in st.session_state:
            st.session_state.story_index = 0
            # Reduced to just 4 story parts
            st.session_state.story_content = [
                "You wake up and find yourself quantized. One moment you were human, the next—poof!—you are a qubit...the simplest quantum system possible. And what's more, the quantumland has trapped you in a Bloch sphere, a perfect geometric prison. The curved surface stretches in all directions, its translucent blue walls pulsing with vague symbols that look like matrices.",

                "But wait a second, you see a door on the sphere—surely an anomaly that shouldn't exist in this geometrically perfect prison—and you wonder maybe it will get you out? But how to reach the door? You try to move, but the rules are different here. Linear motion is impossible. You can only rotate, superpose, dephase....",

                "Suddenly, you hear a voice that seems to come from everywhere and nowhere at once. 'Ah, a new consciousness enters the quantum realm,' the voice resonates, its tone both amused and curious. 'Lost between states, are we? How delightfully uncertain!' The voice presents you with riddles and says 'finally a spin to my tale! If you wish to get back to your hooman form, you must solve some puzzles first!'",

                "A control panel materializes before you—gates labeled with strange symbols: X, Y, Z, H, S, T. You recognize them somehow as quantum gates—tools to manipulate your very state of being. 'Choose wisely,' the voice purrs. 'The wrong transformation might scatter your existence. But the right one...' The voice trails off into a chuckle that sounds like static interference.",
                
                "But before you can learn how top navigate this world, you must first learn what each of the moves do! Solve the riddles by finding the correct gate match! But remember, it is important to figure out how many times you must apply a gate to come back to yourself! You realize this will surely help you later on even if you applied the wrong gate! You can undo the action by knowing precisely how many times you must apply the same gate!"
            ]

        if 'current_riddle' not in st.session_state:
            st.session_state.current_riddle = 0
            st.session_state.riddles = [
                {
                    "text": "Flip me over, flip me back, apply me twice and there's no lack. What gate am I that makes this track?",
                    "answer": "X", "hint": "This gate flips |0⟩ to |1⟩ and vice versa."},
                {
                    "text": "Around the Y-axis I'll rotate, apply me twice and seal your fate. Back to the start, no change, no wait. What gate am I on this quantum date?",
                    "answer": "Y", "hint": "This gate rotates around the Y-axis of the Bloch sphere."},
                {
                    "text": "I change your phase but not your chance, apply me two times to complete this dance. What gate am I in this quantum trance?",
                    "answer": "Z", "hint": "This gate adds a phase flip to |1⟩ states."},
                {
                    "text": "I'm equal parts X, Y, and Z, apply me twice and you will see, we're back where we started, identity! What superposition gate could I be?",
                    "answer": "H", "hint": "This gate creates superpositions of states."},
                {
                    "text": "A quarter turn around the Z, apply me four times, if you please. Only then will your qubit be at ease. What gate am I with such expertise?",
                    "answer": "S", "hint": "This gate rotates by π/2 around the Z-axis."},
                {
                    "text": "One-eighth of a full rotation, apply me eight times for restoration. Patience is key for this quantum equation. What gate am I in this calculation?",
                    "answer": "T", "hint": "This gate is the π/8 gate, rotating by π/4 around the Z-axis."}
            ]
            st.session_state.riddle_solved = False
            st.session_state.enable_gates = False
            st.session_state.show_hint = False
            st.session_state.attempts = 0
            st.session_state.score = 0
            st.session_state.gate_effects = {
                'X': "You feel yourself flip along the X-axis, inverting your state completely!",
                'Y': "You spin around the Y-axis, experiencing a phase shift as your reality inverts!",
                'Z': "A phase flip ripples through your quantum essence along the Z-axis!",
                'H': "You're thrown into superposition, existing in multiple states simultaneously!",
                'S': "You rotate a quarter turn, feeling the phase of your existence shift!",
                'T': "An eighth of a rotation twists your quantum nature ever so slightly!"
            }

        # Initialize state if not already done
        if 'initial_state_rho1' not in st.session_state:
            rho_in = one_qubit_state(0.1, 0.2, 0.5)
            st.session_state.initial_state_rho = rho_in
            st.session_state.initial_state_rho1 = rho_1(rho_in)
            st.session_state.gate_history1 = []
            st.session_state.state_rho = rho_in
            st.session_state.state_rho1 = st.session_state.initial_state_rho1

            # Calculate final state once
            initial_state = st.session_state.initial_state_rho1
            st.session_state.final_state_rho1 = apply_s_gate(apply_z_gate(apply_y_gate(apply_x_gate(initial_state))))

        # Add custom CSS
        st.markdown("""
        <style>
        .story-container {
            background-color: rgba(20, 20, 60, 0.8);
            color:#f0e6ff;
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0;
            border: 2px solid #4d79ff;
            font-family: 'Share Tech Mono', monospace;
            box-shadow: 0 0 15px rgba(77, 121, 255, 0.5);
            }
        .riddle-container {
            background-color: rgba(50, 0, 80, 0.8);
            color: #f0e6ff;
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0 25px 0;
            border: 2px solid #9966ff;
            font-family: 'Share Tech Mono', monospace;
            text-align: center;
            box-shadow: 0 0 15px rgba(153, 102, 255, 0.5);
        }
        .gate-effect {
            background-color: rgba(30, 40, 90, 0.8);
            color: #d1e0ff;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            border: 1px solid #6699ff;
            font-style: italic;
        }
        .translucent-container {
            background-color: rgba(240, 240, 255, 0.2);
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        </style>
        """, unsafe_allow_html=True)

        # Debug information
        st.sidebar.write(f"Story index: {st.session_state.story_index}")
        st.sidebar.write(f"Story length: {len(st.session_state.story_content)}")
        st.sidebar.write(f"Gates enabled: {st.session_state.enable_gates}")
        st.sidebar.write(f"Current riddle: {st.session_state.current_riddle}")
        st.sidebar.write(f"Feedback active: {st.session_state.feedback_active}")
        st.sidebar.write(f"Feedback type: {st.session_state.feedback_type}")

        # Content area
        if st.session_state.story_index < len(st.session_state.story_content):
            # Show story
            st.markdown(f"""
            <div class="story-container">
                <p style="font-size: 1.1em; line-height: 1.5;">{st.session_state.story_content[st.session_state.story_index]}</p>
            </div>
            """, unsafe_allow_html=True)

            # Continue button
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("Continue", key="story_next"):
                    st.session_state.story_index += 1
                    if st.session_state.story_index >= len(st.session_state.story_content):
                        st.session_state.enable_gates = True
                    st.rerun()
        else:
            # Story is complete, show riddles and game
            if st.session_state.current_riddle < len(st.session_state.riddles):
                # Show current riddle
                current_riddle = st.session_state.riddles[st.session_state.current_riddle]
                st.markdown(f"""
                <div class="riddle-container">
                    <h3 style="color: #d9b3ff;">Quantum Riddle {st.session_state.current_riddle + 1}</h3>
                    <p style="font-size: 1.2em; line-height: 1.5; font-style: italic;">
                        {current_riddle['text']}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Hint button
                hint_col1, hint_col2, hint_col3 = st.columns([1, 1, 1])
                with hint_col2:
                    if st.button("Get Hint", key=f"hint_{st.session_state.current_riddle}"):
                        st.session_state.show_hint = True

                if st.session_state.show_hint:
                    st.info(f"Hint: {current_riddle['hint']}")

            # Create placeholders for feedback messages
            feedback_placeholder = st.empty()
            success_feedback_placeholder = st.empty()
            info_feedback_placeholder = st.empty()

            # Display feedback messages if active (MUST BE BEFORE GATE PROCESSING)
            if st.session_state.feedback_active:
                if st.session_state.feedback_type == "success":
                    with success_feedback_placeholder:
                        st.success(st.session_state.feedback_message)

                    # Add a "Get ready for next riddle" message if not all riddles are solved
                    if st.session_state.current_riddle < len(st.session_state.riddles):
                        with info_feedback_placeholder:
                            st.info("Get ready for the next riddle!")
                elif st.session_state.feedback_type == "warning":
                    with success_feedback_placeholder:
                        st.warning(st.session_state.feedback_message)

            # Display gate effect if it exists
            if 'gate_effect_message' in st.session_state and st.session_state.gate_effect_message:
                with feedback_placeholder:
                    st.markdown(f"""
                    <div class="gate-effect">
                        {st.session_state.gate_effect_message}
                    </div>
                    """, unsafe_allow_html=True)

            # Show Bloch sphere
            final_bloch_vector1 = bloch_vector(st.session_state.final_state_rho1)
            current_bloch_vector1 = bloch_vector(st.session_state.state_rho1)
            plot_bloch_sphere(current_bloch_vector1, final_bloch_vector1, 'blues', "Bloch Sphere 1")

            # Gate controls
            st.write("## Apply gate to *your-qubit-self*:")
            st.markdown('<div class="translucent-container">', unsafe_allow_html=True)

            # Collect gate inputs
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                x_button = st.button('X Gate (Qubit 1)')
                y_button = st.button('Y Gate (Qubit 1)')
                z_button = st.button('Z Gate (Qubit 1)')

            with subcol2:
                h_button = st.button('H Gate (Qubit 1)')
                s_button = st.button('S Gate (Qubit 1)')
                t_button = st.button('T Gate (Qubit 1)')

            st.markdown('</div>', unsafe_allow_html=True)

            # Process gate clicks
            gate_applied = None
            correct_gate = st.session_state.riddles[st.session_state.current_riddle][
                "answer"] if st.session_state.current_riddle < len(st.session_state.riddles) else None

            if x_button:
                gate = pauli_x()
                st.session_state.state_rho = np.kron(gate, np.eye(2)) @ st.session_state.state_rho @ np.kron(
                    gate.conj().T, np.eye(2))
                st.session_state.gate_history1.append('X')
                st.session_state.state_rho1 = rho_1(st.session_state.state_rho)
                gate_applied = 'X'

            elif y_button:
                gate = pauli_y()
                st.session_state.state_rho = np.kron(gate, np.eye(2)) @ st.session_state.state_rho @ np.kron(
                    gate.conj().T, np.eye(2))
                st.session_state.gate_history1.append('Y')
                st.session_state.state_rho1 = rho_1(st.session_state.state_rho)
                gate_applied = 'Y'

            elif z_button:
                gate = pauli_z()
                st.session_state.state_rho = np.kron(gate, np.eye(2)) @ st.session_state.state_rho @ np.kron(
                    gate.conj().T, np.eye(2))
                st.session_state.gate_history1.append('Z')
                st.session_state.state_rho1 = rho_1(st.session_state.state_rho)
                gate_applied = 'Z'

            elif h_button:
                gate = hadamard()
                st.session_state.state_rho = np.kron(gate, np.eye(2)) @ st.session_state.state_rho @ np.kron(
                    gate.conj().T, np.eye(2))
                st.session_state.gate_history1.append('H')
                st.session_state.state_rho1 = rho_1(st.session_state.state_rho)
                gate_applied = 'H'

            elif s_button:
                gate = s_gate()
                st.session_state.state_rho = np.kron(gate, np.eye(2)) @ st.session_state.state_rho @ np.kron(
                    gate.conj().T, np.eye(2))
                st.session_state.gate_history1.append('S')
                st.session_state.state_rho1 = rho_1(st.session_state.state_rho)
                gate_applied = 'S'

            elif t_button:
                gate = t_gate()
                st.session_state.state_rho = np.kron(gate, np.eye(2)) @ st.session_state.state_rho @ np.kron(
                    gate.conj().T, np.eye(2))
                st.session_state.gate_history1.append('T')
                st.session_state.state_rho1 = rho_1(st.session_state.state_rho)
                gate_applied = 'T'

            # Handle gate application and riddle solving
            if gate_applied:
                # Set gate effect message
                st.session_state.gate_effect_message = st.session_state.gate_effects[gate_applied]

                # Check if gate is correct
                if gate_applied == correct_gate:
                    st.session_state.feedback_active = True
                    st.session_state.feedback_message = f"Correct! The {gate_applied} gate was the answer."
                    st.session_state.feedback_type = "success"
                    st.session_state.score += max(10 - st.session_state.attempts * 2, 1)
                    st.session_state.current_riddle += 1
                    st.session_state.attempts = 0
                    st.session_state.show_hint = False

                    # Check if all riddles are solved
                    if st.session_state.current_riddle >= len(st.session_state.riddles):
                        st.session_state.feedback_message += " You've solved all the riddles! The door on the Bloch sphere begins to glow...Scroll to the bottom to unlock the next level"
                        st.session_state.all_complete = True
                else:
                    st.session_state.feedback_active = True
                    st.session_state.feedback_message = f"That wasn't the right gate for this riddle. Try again!"
                    st.session_state.feedback_type = "warning"
                    st.session_state.attempts += 1

                # Show balloons if all complete
                if hasattr(st.session_state, 'all_complete') and st.session_state.all_complete:
                    st.balloons()

                # Trigger page rerun
                st.rerun()

            # Display state info
            display_rho(st.session_state.state_rho1)
            update_system(current_bloch_vector1, st.session_state.state_rho1, st.session_state.gate_history1, 'pink',
                          'circuit1')

            # Density matrix visualization
            #st.subheader("Density Matrix Real Parts")
            #fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
            #rho_real = np.real(st.session_state.state_rho)
            #cax1 = ax1.imshow(rho_real, cmap="BuPu", interpolation="nearest")
            #fig.colorbar(cax1, ax=ax1)
            #ax1.set_title("Real Part of Density Matrix")
            #plt.tight_layout()
            #st.pyplot(fig)

            # Level completion check
            if st.session_state.current_riddle >= len(st.session_state.riddles) or st.session_state.score >= 50:
                if st.button("Click when done!"):
                    score = 50
                    if level(score):
                        level_transition()

                if st.button("Take me to next level"):
                    st.session_state.current_level = 2
                    st.rerun()

    if st.session_state.current_level == 2:
        st.header("Level 2: Now find a way out of *your-qubit-self!*")

        # Initialize feedback state variables
        if 'level1b_feedback_active' not in st.session_state:
            st.session_state.level1b_feedback_active = False
            st.session_state.level1b_feedback_message = ""
            st.session_state.level1b_feedback_type = ""
            st.session_state.level1b_gate_effect_message = ""

        # Define your rho_1 and rho_2 functions
        def rho_1(rho_2qubit):
            return np.array([[rho_2qubit[0, 0] + rho_2qubit[1, 1], rho_2qubit[0, 2] + rho_2qubit[1, 3]],
                             [rho_2qubit[2, 0] + rho_2qubit[3, 1], rho_2qubit[2, 2] + rho_2qubit[3, 3]]])

        def rho_2(rho_2qubit):
            return np.array([[rho_2qubit[2, 2] + rho_2qubit[0, 0], rho_2qubit[2, 3] + rho_2qubit[0, 1]],
                             [rho_2qubit[3, 2] + rho_2qubit[1, 0], rho_2qubit[3, 3] + rho_2qubit[1, 1]]])

        # Story and riddle state initialization
        if 'level1b_story_index' not in st.session_state:
            st.session_state.level1b_story_index = 0
            # Reduced to just 4 story parts
            st.session_state.level1b_story_content = [
                "The voice returns, this time with a hint of mischief, 'Ah, now that you know how to move around, lets see you try to get out!'",
                "''See that gate, don't you? Lets see of you can escape this Bloch I have created for you!",
                "'Oh and don't you worry,' the voice adds, 'there will be puzzles guiding you all along the way...'",
                "'But remember, any mistake you make you must undo the actions by coming back to the starting state else you may lose yourself in the infinite possibility mess!'"
            ]

        if 'level1b_current_riddle' not in st.session_state:
            st.session_state.level1b_current_riddle = 0
            st.session_state.level1b_riddles = [
                {
                    "text": "Turn 1 into 0 and 0 into 1 and you will be one step closer to being won ",
                    "answer": "X", "hint": "It's the first gate you learned about!"},
                {
                    "text": "Now lets rotate about the Y, do not ask me why!",
                    "answer": "Y", "hint": "The answer is in the question"},
                {
                    "text": "Without changing your chance, lets do the phase dance",
                    "answer": "Z", "hint": "This gate adds a phase to the |1⟩ states"},
                {
                    "text": "Finally pass through the gate that takes half the amount of times as the longest gate to get back to the original state",
                    "answer": "S", "hint": "T gate takes 8 turns to get back to identity so what gate am I?"}
            ]
            st.session_state.level1b_riddle_solved = False
            st.session_state.level1b_enable_gates = False
            st.session_state.level1b_show_hint = False
            st.session_state.level1b_attempts = 0
            st.session_state.level1b_score = 0
            st.session_state.level1b_gate_effects = {
                'X': "You feel yourself flip along the X-axis, inverting your state completely!",
                'Y': "You spin around the Y-axis, experiencing a phase shift as your reality inverts!",
                'Z': "A phase flip ripples through your quantum essence along the Z-axis!",
                'H': "You're thrown into superposition, existing in multiple states simultaneously!",
                'S': "You rotate a quarter turn, feeling the phase of your existence shift!",
                'T': "An eighth of a rotation twists your quantum nature ever so slightly!"
            }

        # Initialize state if not already done
        if 'initial_state_rho1' not in st.session_state:
            rho_in = one_qubit_state(0.1, 0.2, 0.5)
            st.session_state.initial_state_rho = rho_in
            st.session_state.initial_state_rho1 = rho_1(rho_in)
            st.session_state.gate_history1 = []
            st.session_state.state_rho = rho_in
            st.session_state.state_rho1 = st.session_state.initial_state_rho1

            # Calculate final state once
            initial_state = st.session_state.initial_state_rho1
            st.session_state.final_state_rho1 = apply_s_gate(apply_z_gate(apply_y_gate(apply_x_gate(initial_state))))

        # Add custom CSS
        st.markdown("""
        <style>
        .story-container {
            background-color: rgba(20, 20, 60, 0.8);
            color:#f0e6ff;
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0;
            border: 2px solid #4d79ff;
            font-family: 'Share Tech Mono', monospace;
            box-shadow: 0 0 15px rgba(77, 121, 255, 0.5);
            }
        .riddle-container {
            background-color: rgba(50, 0, 80, 0.8);
            color: #f0e6ff;
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0 25px 0;
            border: 2px solid #9966ff;
            font-family: 'Share Tech Mono', monospace;
            text-align: center;
            box-shadow: 0 0 15px rgba(153, 102, 255, 0.5);
        }
        .gate-effect {
            background-color: rgba(30, 40, 90, 0.8);
            color: #d1e0ff;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            border: 1px solid #6699ff;
            font-style: italic;
        }
        .translucent-container {
            background-color: rgba(240, 240, 255, 0.2);
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        </style>
        """, unsafe_allow_html=True)

        # Debug information
        st.sidebar.write(f"Story index: {st.session_state.level1b_story_index}")
        st.sidebar.write(f"Story length: {len(st.session_state.level1b_story_content)}")
        st.sidebar.write(f"Gates enabled: {st.session_state.level1b_enable_gates}")
        st.sidebar.write(f"Current riddle: {st.session_state.level1b_current_riddle}")

        # Fix missing variable reference
        if 'level1b_feedback_active' in st.session_state:
            st.sidebar.write(f"Feedback active: {st.session_state.level1b_feedback_active}")
            if 'level1b_feedback_type' in st.session_state:
                st.sidebar.write(f"Feedback type: {st.session_state.level1b_feedback_type}")

        # Content area
        if st.session_state.level1b_story_index < len(st.session_state.level1b_story_content):
            # Show story
            st.markdown(f"""
            <div class="story-container">
                <p style="font-size: 1.1em; line-height: 1.5;">{st.session_state.level1b_story_content[st.session_state.level1b_story_index]}</p>
            </div>
            """, unsafe_allow_html=True)

            # Continue button
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("Continue", key="story_next"):
                    st.session_state.level1b_story_index += 1
                    if st.session_state.level1b_story_index >= len(st.session_state.level1b_story_content):
                        st.session_state.level1b_enable_gates = True
                    st.rerun()
        else:
            # Story is complete, show riddles and game
            if st.session_state.level1b_current_riddle < len(st.session_state.level1b_riddles):
                # Show current riddle
                level1b_current_riddle = st.session_state.level1b_riddles[st.session_state.level1b_current_riddle]
                st.markdown(f"""
                <div class="riddle-container">
                    <h3 style="color: #d9b3ff;">Quantum Riddle {st.session_state.level1b_current_riddle + 1}</h3>
                    <p style="font-size: 1.2em; line-height: 1.5; font-style: italic;">
                        {level1b_current_riddle['text']}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Hint button
                hint_col1, hint_col2, hint_col3 = st.columns([1, 1, 1])
                with hint_col2:
                    if st.button("Get Hint", key=f"hint_{st.session_state.level1b_current_riddle}"):
                        st.session_state.level1b_show_hint = True

                if st.session_state.level1b_show_hint:
                    st.info(f"Hint: {level1b_current_riddle['hint']}")

            # Create placeholders for feedback messages
            level1b_feedback_placeholder = st.empty()
            level1b_success_feedback_placeholder = st.empty()
            level1b_info_feedback_placeholder = st.empty()

            # Display feedback messages if active (MUST BE BEFORE GATE PROCESSING)
            if 'level1b_feedback_active' in st.session_state and st.session_state.level1b_feedback_active:
                if 'level1b_feedback_type' in st.session_state and st.session_state.level1b_feedback_type == "success":
                    with level1b_success_feedback_placeholder:
                        st.success(st.session_state.level1b_feedback_message)

                    # Add a "Get ready for next riddle" message if not all riddles are solved
                    if st.session_state.level1b_current_riddle < len(st.session_state.level1b_riddles):
                        with level1b_info_feedback_placeholder:
                            st.info("Get ready for the next riddle!")
                elif 'level1b_feedback_type' in st.session_state and st.session_state.level1b_feedback_type == "warning":
                    with level1b_success_feedback_placeholder:
                        st.warning(st.session_state.level1b_feedback_message)

            # Display gate effect if it exists
            if 'level1b_gate_effect_message' in st.session_state and st.session_state.level1b_gate_effect_message:
                with level1b_feedback_placeholder:
                    st.markdown(f"""
                    <div class="gate-effect">
                        {st.session_state.level1b_gate_effect_message}
                    </div>
                    """, unsafe_allow_html=True)

            # Show Bloch sphere
            final_bloch_vector1 = bloch_vector(st.session_state.final_state_rho1)
            current_bloch_vector1 = bloch_vector(st.session_state.state_rho1)
            plot_bloch_sphere(current_bloch_vector1, final_bloch_vector1, 'blues', "Bloch Sphere 1")

            # Gate controls
            st.write("## Apply gate to *your-qubit-self*:")
            st.markdown('<div class="translucent-container">', unsafe_allow_html=True)

            # Collect gate inputs
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                x_button = st.button('X Gate (Qubit 1)')
                y_button = st.button('Y Gate (Qubit 1)')
                z_button = st.button('Z Gate (Qubit 1)')

            with subcol2:
                h_button = st.button('H Gate (Qubit 1)')
                s_button = st.button('S Gate (Qubit 1)')
                t_button = st.button('T Gate (Qubit 1)')

            with st.expander("Gate controls"):
                st.markdown("""
                X: This gate flips |0⟩ to |1⟩ and vice versa.
                Y: This gate rotates around the Y-axis of the Bloch sphere.
                Z: This gate adds a phase flip to |1⟩ states.
                H: Equal part X,Y and Z, this gate creates superpositions of states.
                S: This gate rotates by π/2 around the Z-axis. It takes 4 turns to get back to your original self.
                T: This gate is the π/8 gate, rotating by π/4 around the Z-axis.
                """)

            st.markdown('</div>', unsafe_allow_html=True)

            # Process gate clicks
            gate_applied = None
            correct_gate = st.session_state.level1b_riddles[st.session_state.level1b_current_riddle][
                "answer"] if st.session_state.level1b_current_riddle < len(st.session_state.level1b_riddles) else None

            if x_button:
                gate = pauli_x()
                st.session_state.state_rho = np.kron(gate, np.eye(2)) @ st.session_state.state_rho @ np.kron(
                    gate.conj().T, np.eye(2))
                st.session_state.gate_history1.append('X')
                st.session_state.state_rho1 = rho_1(st.session_state.state_rho)
                gate_applied = 'X'

            elif y_button:
                gate = pauli_y()
                st.session_state.state_rho = np.kron(gate, np.eye(2)) @ st.session_state.state_rho @ np.kron(
                    gate.conj().T, np.eye(2))
                st.session_state.gate_history1.append('Y')
                st.session_state.state_rho1 = rho_1(st.session_state.state_rho)
                gate_applied = 'Y'

            elif z_button:
                gate = pauli_z()
                st.session_state.state_rho = np.kron(gate, np.eye(2)) @ st.session_state.state_rho @ np.kron(
                    gate.conj().T, np.eye(2))
                st.session_state.gate_history1.append('Z')
                st.session_state.state_rho1 = rho_1(st.session_state.state_rho)
                gate_applied = 'Z'

            elif h_button:
                gate = hadamard()
                st.session_state.state_rho = np.kron(gate, np.eye(2)) @ st.session_state.state_rho @ np.kron(
                    gate.conj().T, np.eye(2))
                st.session_state.gate_history1.append('H')
                st.session_state.state_rho1 = rho_1(st.session_state.state_rho)
                gate_applied = 'H'

            elif s_button:
                gate = s_gate()
                st.session_state.state_rho = np.kron(gate, np.eye(2)) @ st.session_state.state_rho @ np.kron(
                    gate.conj().T, np.eye(2))
                st.session_state.gate_history1.append('S')
                st.session_state.state_rho1 = rho_1(st.session_state.state_rho)
                gate_applied = 'S'

            elif t_button:
                gate = t_gate()
                st.session_state.state_rho = np.kron(gate, np.eye(2)) @ st.session_state.state_rho @ np.kron(
                    gate.conj().T, np.eye(2))
                st.session_state.gate_history1.append('T')
                st.session_state.state_rho1 = rho_1(st.session_state.state_rho)
                gate_applied = 'T'

            # Handle gate application and riddle solving
            if gate_applied:
                # Set gate effect message
                st.session_state.level1b_gate_effect_message = st.session_state.level1b_gate_effects[gate_applied]

                # Check if gate is correct
                if gate_applied == correct_gate:
                    st.session_state.level1b_feedback_active = True
                    st.session_state.level1b_feedback_message = f"Correct! The {gate_applied} gate was the answer."
                    st.session_state.level1b_feedback_type = "success"
                    st.session_state.level1b_score += max(10 - st.session_state.level1b_attempts * 2, 1)
                    st.session_state.level1b_current_riddle += 1
                    st.session_state.level1b_attempts = 0
                    st.session_state.level1b_show_hint = False

                    # Check if all riddles are solved
                    if st.session_state.level1b_current_riddle >= len(st.session_state.level1b_riddles):
                        # Set a flag to show balloons AFTER rerun
                        st.session_state.level1b_show_balloons = True
                        st.session_state.level1b_feedback_message += " You've solved all the riddles! The door on the Bloch sphere begins to glow...Scroll to the bottom to unlock the next level"
                        st.session_state.level1b_all_complete = True
                else:
                    st.session_state.level1b_feedback_active = True
                    st.session_state.level1b_feedback_message = f"That wasn't the right gate for this riddle. Try again!"
                    st.session_state.level1b_feedback_type = "warning"
                    st.session_state.level1b_attempts += 1

                # Trigger page rerun
                st.rerun()

            # Put this OUTSIDE any conditional blocks, at the top level of your script
            if hasattr(st.session_state, 'level1b_show_balloons') and st.session_state.level1b_show_balloons:
                st.balloons()
                # Clear the flag so balloons don't keep showing
                st.session_state.level1b_show_balloons = False

            # Display state info
            display_rho(st.session_state.state_rho1)
            update_system(current_bloch_vector1, st.session_state.state_rho1, st.session_state.gate_history1, 'pink',
                          'circuit1')

            # Density matrix visualization
            #st.subheader("Density Matrix Real Parts")
            #fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
            #rho_real = np.real(st.session_state.state_rho)
            #cax1 = ax1.imshow(rho_real, cmap="BuPu", interpolation="nearest")
            #fig.colorbar(cax1, ax=ax1)
            #ax1.set_title("Real Part of Density Matrix")
            #plt.tight_layout()
            #st.pyplot(fig)

            # Level completion check
            if st.session_state.level1b_current_riddle >= len(
                    st.session_state.level1b_riddles) or st.session_state.level1b_score >= 50:
                if st.button("Click when done!"):
                    score = 50
                    if level(score):
                        level_transition()

                if st.button("Take me to next level"):
                    st.session_state.current_level = 3
                    st.rerun()

    elif st.session_state.current_level == 3:  # or should this be level 3?

        st.header("Level 3: Quantum Rotations")

        def rho_1(rho_2qubit):

            return np.array([[rho_2qubit[0, 0] + rho_2qubit[1, 1], rho_2qubit[0, 2] + rho_2qubit[1, 3]],

                             [rho_2qubit[2, 0] + rho_2qubit[3, 1], rho_2qubit[2, 2] + rho_2qubit[3, 3]]])

        def rho_2(rho_2qubit):

            return np.array([[rho_2qubit[2, 2] + rho_2qubit[0, 0], rho_2qubit[2, 3] + rho_2qubit[0, 1]],

                             [rho_2qubit[3, 2] + rho_2qubit[1, 0], rho_2qubit[3, 3] + rho_2qubit[1, 1]]])

        # Story and riddle state initialization

        if 'level2_story_index' not in st.session_state:
            st.session_state.level2_story_index = 0

            st.session_state.level2_story_content = [

                "Having mastered the basic quantum gates, you face a new challenge. The voice speaks again: 'Well done, quantum traveler. But discrete operations can only take you so far...'",

                "You notice the control panel has transformed. The discrete gate buttons have been replaced by continuous sliders. 'Welcome to the world of parameterized quantum rotations,' the voice continues.",

                "These rotation channels allow you to apply transformations with varying strength, unlike the fixed rotations of the basic gates. The angle parameter gives you precise control over your quantum state.",

                "'Use the Rx, Ry, and Rz channels to navigate the quantum realm with greater finesse. Adjust the sliders to control the rotation angle, then apply the channel to see its effect on your qubit state.'"

            ]

        # Initialize state if not already done

        rho_2qubit_initial = one_qubit_state(0.1, 0.2, 0.5)

        if 'initial_state_rho' not in st.session_state:
            st.session_state.initial_state_rho = rho_2qubit_initial

            st.session_state.initial_state_rho1 = rho_1(rho_2qubit_initial)

            st.session_state.initial_state_rho2 = rho_2(rho_2qubit_initial)

            st.session_state.state_rho = rho_2qubit_initial

            st.session_state.state_rho1 = rho_1(rho_2qubit_initial)

            st.session_state.state_rho2 = rho_2(rho_2qubit_initial)

        # Initialize empty gate histories

        if 'gate_history1' not in st.session_state:
            st.session_state.gate_history1 = []

        if 'gate_history2' not in st.session_state:
            st.session_state.gate_history2 = []

        # Define the target final state

        if 'final_state_rho1' not in st.session_state:
            # Make sure initial_state is properly defined before using it

            initial_state = st.session_state.initial_state_rho1  # Use the already initialized state

            st.session_state.final_state_rho1 = apply_z_gate(apply_z_gate(apply_y_gate(apply_x_gate(initial_state))))



        # Add custom CSS for story display

        st.markdown("""

        <style>

        .story-container {

            background-color: rgba(20, 20, 60, 0.8);

            color: #f0e6ff;

            padding: 20px;

            border-radius: 10px;

            margin: 15px 0;

            border: 2px solid #4d79ff;

            font-family: 'Share Tech Mono', monospace;

            box-shadow: 0 0 15px rgba(77, 121, 255, 0.5);

        }

        .channel-container {

            background-color: rgba(50, 10, 70, 0.6);

            padding: 20px;

            border-radius: 15px;

            margin: 15px 0;

            border: 2px solid #FF9900;

            box-shadow: 0 0 15px rgba(255, 153, 0, 0.3);

        }

        .channel-label {

            color: #9FFFCB;

            font-family: 'Share Tech Mono', monospace;

            font-size: 1.1em;

            margin-bottom: 10px;

        }

        </style>

        """, unsafe_allow_html=True)

        # Create page layout

        col1, col2, col3 = st.columns([1, 6, 1])

        with col2:

            # Display story if not finished

            if st.session_state.level2_story_index < len(st.session_state.level2_story_content):

                st.markdown(f"""
                <div class="story-container">
                    <p style="font-size: 1.1em; line-height: 1.5;">{st.session_state.level2_story_content[st.session_state.level2_story_index]}</p>
                </div>
                """, unsafe_allow_html=True)

                # Next button for story

                col1, col2, col3 = st.columns([1, 1, 1])

                with col2:

                    if st.button("Continue", key="level2_story_next"):
                        st.session_state.level2_story_index += 1
                        st.rerun()

            else:

                # Story is complete, show the main interface

                #st.header("Quantum Rotation Channels")

                # Create placeholders for visualizations that will be updated after button clicks

                bloch_placeholder = st.empty()

                rho_placeholder = st.empty()

                circuit_placeholder = st.empty()

                # Current and target state visualization

                final_bloch_vector1 = bloch_vector(st.session_state.final_state_rho1)

                current_bloch_vector1 = bloch_vector(st.session_state.state_rho1)

                with bloch_placeholder:

                    plot_bloch_sphere(current_bloch_vector1, final_bloch_vector1, 'blues', "Bloch Sphere 1")

                # Quantum channel controls - ONE ROW WITH THREE COLUMNS

                st.markdown('<div class="channel-container">', unsafe_allow_html=True)

                st.markdown('<p class="channel-label">Apply a quantum rotation:</p>', unsafe_allow_html=True)

                # Create a single row with three columns

                rx_col, ry_col, rz_col = st.columns(3)

                # X-rotation channel

                with rx_col:

                    st.markdown('<div style="border-left: 4px solid #FF5555; padding-left: 10px; margin: 15px 0;">',
                                unsafe_allow_html=True)

                    st.markdown('<p style="color: #FF9999; font-weight: bold;">X-Rotation Channel</p>',
                                unsafe_allow_html=True)

                    px1_bar = st.slider("X-axis angle (π)", 0.0, 2.0, 0.1, step=0.05)

                    rx_button = st.button('Apply Rx Channel')

                    st.markdown('</div>', unsafe_allow_html=True)

                # Y-rotation channel

                with ry_col:

                    st.markdown('<div style="border-left: 4px solid #55FF55; padding-left: 10px; margin: 15px 0;">',
                                unsafe_allow_html=True)

                    st.markdown('<p style="color: #99FF99; font-weight: bold;">Y-Rotation Channel</p>',
                                unsafe_allow_html=True)

                    py1_bar = st.slider("Y-axis angle (π)", 0.0, 2.0, 0.1, step=0.05)

                    ry_button = st.button('Apply Ry Channel')

                    st.markdown('</div>', unsafe_allow_html=True)

                # Z-rotation channel

                with rz_col:

                    st.markdown('<div style="border-left: 4px solid #5555FF; padding-left: 10px; margin: 15px 0;">',
                                unsafe_allow_html=True)

                    st.markdown('<p style="color: #9999FF; font-weight: bold;">Z-Rotation Channel</p>',
                                unsafe_allow_html=True)

                    pz1_bar = st.slider("Z-axis angle (π)", 0.0, 2.0, 0.1, step=0.05)

                    rz_button = st.button('Apply Rz Channel')

                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

                # Information about channels

                with st.expander("Learn about Quantum Rotation Channels"):

                    st.markdown("""

                    **Rx(θ)**: Rotation around the X-axis by angle θ. When θ = π, this is equivalent to the X gate.


                    **Ry(θ)**: Rotation around the Y-axis by angle θ. When θ = π, this is equivalent to the Y gate.


                    **Rz(θ)**: Rotation around the Z-axis by angle θ. When θ = π, this is equivalent to the Z gate.


                    These parameterized rotation gates give you precise control over quantum rotations, allowing you to navigate the Bloch sphere with greater flexibility than the standard gates.

                    """)

                # Process button clicks and update state

                if rx_button:

                    gate = r_x_rotation(px1_bar * np.pi)  # Convert slider value to radians

                    st.session_state.state_rho = np.kron(gate, np.eye(2)) @ st.session_state.state_rho @ np.kron(

                        gate.conj().T, np.eye(2))

                    st.session_state.gate_history1.append(f'Rx({px1_bar:.2f}π)')

                    st.session_state.state_rho1 = rho_1(st.session_state.state_rho)

                    st.rerun()


                elif ry_button:

                    gate = r_y_rotation(py1_bar * np.pi)  # Convert slider value to radians

                    st.session_state.state_rho = np.kron(gate, np.eye(2)) @ st.session_state.state_rho @ np.kron(

                        gate.conj().T, np.eye(2))

                    st.session_state.gate_history1.append(f'Ry({py1_bar:.2f}π)')

                    st.session_state.state_rho1 = rho_1(st.session_state.state_rho)

                    st.rerun()


                elif rz_button:

                    gate = r_z_rotation(pz1_bar * np.pi)  # Convert slider value to radians

                    st.session_state.state_rho = np.kron(gate, np.eye(2)) @ st.session_state.state_rho @ np.kron(

                        gate.conj().T, np.eye(2))

                    st.session_state.gate_history1.append(f'Rz({pz1_bar:.2f}π)')

                    st.session_state.state_rho1 = rho_1(st.session_state.state_rho)

                    st.rerun()

                # Display state information

                with rho_placeholder:

                    display_rho(st.session_state.state_rho1)

                with circuit_placeholder:

                    update_system(current_bloch_vector1, st.session_state.state_rho1,

                                  st.session_state.gate_history1, 'pink', 'circuit1')

                # Density matrix visualization

                #st.subheader("Density Matrix Real Parts")

                # Level completion

                score = 0

                if st.button("Click when done!"):
                    score = 50

                if level(score):
                    level_transition()

                st.button("Take me to next level")

                #fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))

                #rho_real = np.real(st.session_state.state_rho)

                #cax1 = ax1.imshow(rho_real, cmap="BuPu", interpolation="nearest")

                #fig.colorbar(cax1, ax=ax1)

                #ax1.set_title("Real Part of Density Matrix")

                #plt.tight_layout()

                #st.pyplot(fig)

        # Game completion check

    elif st.session_state.current_level == 4:

        st.header("Level 4: Universal Gates")

        # Define your rho_1 and rho_2 functions

        def rho_1(rho_2qubit):

            return np.array([[rho_2qubit[0, 0] + rho_2qubit[1, 1], rho_2qubit[0, 2] + rho_2qubit[1, 3]],

                             [rho_2qubit[2, 0] + rho_2qubit[3, 1], rho_2qubit[2, 2] + rho_2qubit[3, 3]]])

        def rho_2(rho_2qubit):

            return np.array([[rho_2qubit[2, 2] + rho_2qubit[0, 0], rho_2qubit[2, 3] + rho_2qubit[0, 1]],

                             [rho_2qubit[3, 2] + rho_2qubit[1, 0], rho_2qubit[3, 3] + rho_2qubit[1, 1]]])

        # Story and riddle state initialization

        if 'level3_story_index' not in st.session_state:
            st.session_state.level3_story_index = 0
            st.session_state.level3_story_content = [

                "The voice returns, this time with a different tone—more instructive, yet somehow more cryptic. If you must move alone in this Bloch sphere, it intones, you shall not need all these operations. All you need is the universal set of gates.",

                "Before you, three gates illuminate brighter than the others: Ry, Rz, and phase shift gates. They pulse with a different quality of light, as if they're somehow more fundamental than the others.",

                "\"With these alone,\" the voice continues, \"any transformation can be achieved. Any state can be reached. Any door... unlocked.\" You reach toward them, but the voice interrupts with a warning that sends a chill through your quantum state.",

                "\"However,\" it whispers, \"the parameters must be precise. The angles, the phases, the timing—all must be exact. If you do not use the correct variables to build your gate, you may find yourself trapped in an infinite composition.\""

            ]

        # Universal gate challenge setup

        if 'level3_challenge_complete' not in st.session_state:
            st.session_state.level3_challenge_complete = False

        # Initialize state if not already done

        rho_2qubit_initial = one_qubit_state(0.1, 0.1, 0.5)

        if 'level3_initial_state_rho' not in st.session_state:
            st.session_state.level3_initial_state_rho = rho_2qubit_initial

            st.session_state.level3_initial_state_rho1 = rho_1(rho_2qubit_initial)

            st.session_state.level3_initial_state_rho2 = rho_2(rho_2qubit_initial)

            st.session_state.level3_state_rho = rho_2qubit_initial

            st.session_state.level3_state_rho1 = rho_1(rho_2qubit_initial)

            st.session_state.level3_state_rho2 = rho_2(rho_2qubit_initial)

            st.session_state.level3_final_state_rho1 = apply_h_gate(

                apply_y_gate(apply_x_gate(rho_1(rho_2qubit_initial))))

            st.session_state.level3_gate_history1 = []

            st.session_state.level3_attempts = 0

            st.session_state.level3_score = 0

        # Add custom CSS

        st.markdown("""

        <style>

        .story-container {

            background-color: rgba(20, 20, 60, 0.8);

            color: #f0e6ff;

            padding: 20px;

            border-radius: 10px;

            margin: 15px 0;

            border: 2px solid #4d79ff;

            font-family: 'Share Tech Mono', monospace;

            box-shadow: 0 0 15px rgba(77, 121, 255, 0.5);

        }

        .riddle-container {

            background-color: rgba(50, 0, 80, 0.8);

            color: #f0e6ff;

            padding: 20px;

            border-radius: 10px;

            margin: 15px 0 25px 0;

            border: 2px solid #9966ff;

            font-family: 'Share Tech Mono', monospace;

            text-align: center;

            box-shadow: 0 0 15px rgba(153, 102, 255, 0.5);

        }

        .gate-effect {

            background-color: rgba(30, 40, 90, 0.8);

            color: #d1e0ff;

            padding: 15px;

            border-radius: 8px;

            margin: 15px 0;

            border: 1px solid #6699ff;

            font-style: italic;

        }

        .translucent-container {

            background-color: rgba(240, 240, 255, 0.2);

            padding: 20px;

            border-radius: 10px;

            margin: 10px 0;

        }

        </style>

        """, unsafe_allow_html=True)

        # Debug information

        st.sidebar.write(f"Level 3 Story index: {st.session_state.level3_story_index}")

        st.sidebar.write(f"Level 3 Story length: {len(st.session_state.level3_story_content)}")

        st.sidebar.write(f"Level 3 Gate history: {st.session_state.level3_gate_history1}")

        # Content area

        col1, col2, col3 = st.columns([1, 6, 1])

        with col2:

            if st.session_state.level3_story_index < len(st.session_state.level3_story_content):


                st.markdown(f"""
                <div class="story-container">
                    <p style="font-size: 1.1em; line-height: 1.5;">{st.session_state.level3_story_content[st.session_state.level3_story_index]}</p>
                </div>
                """, unsafe_allow_html=True)

                # Continue button

                col1, col2, col3 = st.columns([1, 1, 1])

                with col2:

                    if st.button("Continue", key="level3_story_next"):
                        st.session_state.level3_story_index += 1

                        st.rerun()

            else:

                # Story is complete, show challenge

                if not st.session_state.level3_challenge_complete:
                    st.markdown(f"""

                    <div class="riddle-container">
                        <h3 style="color: #d9b3ff;">Universal Gates Challenge</h3>
                        <p style="font-size: 1.2em; line-height: 1.5;">
                            A general unitary may be written as:
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # For mathematical content, use LaTeX with Streamlit's built-in rendering

                    st.latex(r'''
                    \Large
                    U = \begin{pmatrix}
                    e^{i(\alpha - \beta-\delta)}\cos(\gamma) & -  e^{i(\alpha - \beta+\delta)}\sin(\gamma)\\
                    e^{i(\alpha + \beta-\delta)}\sin(\gamma) & e^{i(\alpha+ \beta+\delta)}\cos(\gamma)
                        \end{pmatrix}
                    ''')

                    st.markdown(f"""

                    <div class="riddle-container">
                        <p style="font-size: 1.2em; line-height: 1.5;">
                            You need to recreate the target state using a sequence of gates in the form:
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.latex(r'''e^{i 2\delta} R_z (2 \gamma) R_y (2 \beta) R_z(\alpha)''')
                    st.markdown(f"""
                    <div class="gate-effect">
                        <p>The exact values you need will appear in the target state. Quantum traveler, all you need is four pi.</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Show feedback placeholder for gate effects

                feedback_placeholder = st.empty()

                # Show Bloch sphere

                final_bloch_vector1 = bloch_vector(st.session_state.level3_final_state_rho1)

                current_bloch_vector1 = bloch_vector(st.session_state.level3_state_rho1)

                plot_bloch_sphere(current_bloch_vector1, final_bloch_vector1, 'blues', "Bloch Sphere - Universal Gates")

                # Add this after plotting the Bloch sphere but before the gate controls section
                st.write("## Progress to Target:")
                if hasattr(st.session_state, 'debug_distance'):
                    # Invert and normalize the distance (closer = more progress)
                    # Max expected distance on Bloch sphere is 2.0
                    progress = max(0, min(1.0, 1.0 - (st.session_state.debug_distance / 2.0)))
                    st.progress(progress)

                    # Add more detailed feedback
                    if progress > 0.95:
                        st.success("Perfect! You're close enough to escape out!")
                    elif progress > 0.9:
                        st.info("Very close! Just a tiny adjustment needed!")
                    elif progress > 0.75:
                        st.info("Getting closer! Keep adjusting the gates.")
                    elif progress > 0.5:
                        st.warning("Making progress, but still some way to go.")
                    else:
                        st.error("Still far from the target state. Try different gate combinations.")

                    # Display numerical distance
                    st.write(f"Distance to target: {st.session_state.debug_distance:.6f}")

                # Gate controls

                st.write("## Apply Universal Gates:")

                st.markdown('<div class="translucent-container">', unsafe_allow_html=True)

                # Collect gate inputs

                subcol1, subcol2, subcol3 = st.columns(3)

                with subcol1:

                    st.markdown('<p style="color: #99FF99; font-weight: bold;">Y-Rotation Gate</p>',
                                unsafe_allow_html=True)

                    py1_bar = st.slider("Ry angle (π radians)", 0.0, 1.0, 0.1, step=0.01, key="level3_ry_slider")

                    apply_ry = st.button('Apply Ry', key="level3_ry_button")

                with subcol2:

                    st.markdown('<p style="color: #FFCC99; font-weight: bold;">Phase Shift Gate</p>',
                                unsafe_allow_html=True)

                    phi1_bar = st.slider("Phase angle (π radians)", 0.0, 1.0, 0.1, step=0.01, key="level3_phi_slider")

                    apply_phi = st.button('Apply Phase shift', key="level3_phi_button")

                with subcol3:

                    st.markdown('<p style="color: #9999FF; font-weight: bold;">Z-Rotation Gate</p>',
                                unsafe_allow_html=True)

                    pz1_bar = st.slider("Rz angle (π radians)", 0.0, 1.0, 0.1, step=0.01, key="level3_rz_slider")

                    apply_rz = st.button('Apply Rz', key="level3_rz_button")

                st.markdown('</div>', unsafe_allow_html=True)

                # Process gate clicks

                gate_applied = None

                gate_effect = None

                if apply_ry:

                    gate = np.array([[np.cos(py1_bar / 2), -np.sin(py1_bar / 2)],
                                          [np.sin(py1_bar / 2), np.cos(py1_bar / 2)]]) # Convert to radians

                    st.session_state.level3_state_rho = np.kron(gate, np.eye(
                        2)) @ st.session_state.level3_state_rho @ np.kron(

                        gate.conj().T, np.eye(2))

                    st.session_state.level3_gate_history1.append(f'Ry({py1_bar:.2f}π)')

                    st.session_state.level3_state_rho1 = rho_1(st.session_state.level3_state_rho)

                    gate_applied = 'Ry'

                    gate_effect = f"You feel yourself rotate {py1_bar:.2f}π radians around the Y-axis, shifting your quantum state!"


                elif apply_phi:

                    gate = np.array([
                                    [1, 0],
                                    [0, np.exp(1j * phi1_bar)]
                                        ]) # Convert to radians

                    st.session_state.level3_state_rho = np.kron(gate, np.eye(
                        2)) @ st.session_state.level3_state_rho @ np.kron(

                        gate.conj().T, np.eye(2))

                    st.session_state.level3_gate_history1.append(f'Φ({phi1_bar:.2f}π)')

                    st.session_state.level3_state_rho1 = rho_1(st.session_state.level3_state_rho)

                    gate_applied = 'Phi'

                    gate_effect = f"A phase shift of {phi1_bar:.2f}π ripples through your quantum essence, altering your state!"


                elif apply_rz:

                    gate = np.array([[np.exp(-1j * pz1_bar / 2), 0],
                                          [0, np.exp(1j * pz1_bar / 2)]])

                    st.session_state.level3_state_rho = np.kron(gate, np.eye(
                        2)) @ st.session_state.level3_state_rho @ np.kron(

                        gate.conj().T, np.eye(2))

                    st.session_state.level3_gate_history1.append(f'Rz({pz1_bar:.2f}π)')

                    st.session_state.level3_state_rho1 = rho_1(st.session_state.level3_state_rho)

                    gate_applied = 'Rz'

                    gate_effect = f"You rotate {pz1_bar:.2f}π radians around the Z-axis, feeling your quantum state transform!"

                # Handle gate application effects

                if gate_applied:

                    with feedback_placeholder:

                        st.markdown(f"""

                        <div class="gate-effect">

                            {gate_effect}

                        </div>

                        """, unsafe_allow_html=True)

                    # Check if we're close to the target state

                    target_vector = final_bloch_vector1

                    current_vector = bloch_vector(st.session_state.level3_state_rho1)

                    # Calculate distance to target

                    distance = np.linalg.norm(target_vector - current_vector)
                    # Add this right after calculating the distance
                    # After calculating distance but before the if statement
                    # After calculating distance but before any rerun
                    st.session_state.debug_distance = distance
                    st.session_state.debug_target = target_vector
                    st.session_state.debug_current = current_vector

                    if distance < 0.1:  # Close enough to target
                        # Make sure this value persists through reruns
                        st.session_state.level3_challenge_complete = True
                        st.session_state.level3_score = 50

                        # Show success message and balloons
                        st.balloons()
                        st.success("You've reached the target state! The door on the Bloch sphere begins to glow...")

                        # Only rerun if we haven't already shown the success
                        if not hasattr(st.session_state, 'success_shown'):
                            st.session_state.success_shown = True
                            st.rerun()
                    else:
                        # Only rerun if we're not at the target
                        st.rerun()

                # Display state info

                display_rho(st.session_state.level3_state_rho1)

                update_system(current_bloch_vector1, st.session_state.level3_state_rho1,
                              st.session_state.level3_gate_history1, 'pink', 'circuit1')

                # Add a hint section

                with st.expander("Need a hint?"):

                    st.markdown("""

                    **Hint 1**: The target state can be reached using a specific sequence of Rz, Ry, and Rzs.


                    **Hint 2**: Try to first align with the correct longitude using Rz, then adjust the latitude with Ry, then fix the final phase with Rz again.


                    **Hint 3**: The parameters needed are multiples of 0.1π. Try values like 0.4π and 0.3π.

                    """)

                # Density matrix visualization

                #st.subheader("Density Matrix Real Parts")

                #fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))

                #rho_real = np.real(st.session_state.level3_state_rho)

                #cax1 = ax1.imshow(rho_real, cmap="BuPu", interpolation="nearest")

                #fig.colorbar(cax1, ax=ax1)

                #ax1.set_title("Real Part of Density Matrix")

                #plt.tight_layout()

                #st.pyplot(fig)

                # Level completion check

                if st.session_state.level3_challenge_complete or st.session_state.level3_score >= 50:

                    if st.button("Click when done!", key="level3_done"):

                        score = 50

                        if level(score):
                            level_transition()

                    if st.button("Take me to next level", key="level3_next"):
                        st.session_state.current_level = 4

                        st.rerun()

        # Game completion check

    elif st.session_state.current_level == 10:
        st.header("Level 5: Entangled Spheres")

        # Define the rho functions for reduced density matrices
        def rho_1(rho_2qubit):
            return np.array([[rho_2qubit[0, 0] + rho_2qubit[1, 1], rho_2qubit[0, 2] + rho_2qubit[1, 3]],
                             [rho_2qubit[2, 0] + rho_2qubit[3, 1], rho_2qubit[2, 2] + rho_2qubit[3, 3]]])

        def rho_2(rho_2qubit):
            return np.array([[rho_2qubit[0, 0] + rho_2qubit[2, 2], rho_2qubit[0, 1] + rho_2qubit[2, 3]],
                             [rho_2qubit[1, 0] + rho_2qubit[3, 2], rho_2qubit[1, 1] + rho_2qubit[3, 3]]])

        # Function to draw two-qubit circuit
        def draw_two_qubit_circuit(gate_history1, gate_history2):
            """
            Draw a two-qubit circuit showing the last 5 gates applied to each qubit.

            Parameters:
            gate_history1 (list): List of gates applied to qubit 1
            gate_history2 (list): List of gates applied to qubit 2
            """
            max_gates = 5
            # Get the last 5 gates or fewer if not enough gates
            truncated_history1 = gate_history1[-max_gates:] if len(gate_history1) > 0 else []
            truncated_history2 = gate_history2[-max_gates:] if len(gate_history2) > 0 else []

            # Calculate the maximum number of gates to show
            max_cols = max(len(truncated_history1), len(truncated_history2), 1)

            # Create the figure
            circuit_fig = go.Figure()

            # Add the qubit wires
            circuit_fig.add_trace(go.Scatter(
                x=[0, max_cols + 1],
                y=[1, 1],
                mode="lines",
                line=dict(width=3, color="black"),
                name="Qubit 1 Wire"
            ))

            circuit_fig.add_trace(go.Scatter(
                x=[0, max_cols + 1],
                y=[0, 0],
                mode="lines",
                line=dict(width=3, color="black"),
                name="Qubit 2 Wire"
            ))

            # Add labels for the qubits
            circuit_fig.add_annotation(
                x=0, y=1,
                text="Qubit 1",
                showarrow=False,
                font=dict(size=14, color="black")
            )

            circuit_fig.add_annotation(
                x=0, y=0,
                text="Qubit 2",
                showarrow=False,
                font=dict(size=14, color="black")
            )

            # Add gates for qubit 1
            for i, gate in enumerate(truncated_history1):
                circuit_fig.add_trace(go.Scatter(
                    x=[i + 1], y=[1],
                    mode="markers+text",
                    marker=dict(symbol="square", size=70, color="pink"),
                    text=[gate],
                    textposition='middle center',
                    textfont=dict(size=25, color="black"),
                    name=f"Qubit 1 Gate {i + 1}"
                ))

            # Add gates for qubit 2
            for i, gate in enumerate(truncated_history2):
                circuit_fig.add_trace(go.Scatter(
                    x=[i + 1], y=[0],
                    mode="markers+text",
                    marker=dict(symbol="square", size=70, color="skyblue"),
                    text=[gate],
                    textposition='middle center',
                    textfont=dict(size=25, color="black"),
                    name=f"Qubit 2 Gate {i + 1}"
                ))

            # Update layout
            circuit_fig.update_layout(
                xaxis=dict(range=[-0.5, max_cols + 1.5], zeroline=False, showticklabels=False),
                yaxis=dict(range=[-0.5, 1.5], zeroline=False, showticklabels=False),
                margin=dict(l=30, r=30, b=30, t=30),
                height=200,
                width=500,
                showlegend=False,
                title="Two-Qubit Circuit History (Last 5 Gates)"
            )

            # Display the circuit - use a unique key based on the lengths of both histories
            st.plotly_chart(circuit_fig, key=f"two_qubit_circuit_{len(gate_history1)}_{len(gate_history2)}")

        # Story initialization
        if 'level4_story_index' not in st.session_state:
            st.session_state.level4_story_index = 0
            st.session_state.level4_story_content = [
                "You input the final sequence of gates, and the door on the Bloch sphere slides open with a resonant hum. Relief washes over your quantum state as you rush through the opening.",

                "But something isn't right. As your consciousness settles, you realize you're still in a Bloch sphere—just a different one. And there, across the curved quantum landscape, is Sonalika, your friend who disappeared a week before you did. Their quantum signature flickers with recognition when they see you.",

                "\"You made it!\" Sonalika calls out, their voice carrying across the quantum void. \"I've been waiting for—\" They stop mid-sentence as both of you suddenly lurch sideways. When you shifted your position, Sonalika moved too—perfectly mirroring your rotation but in the opposite direction.",

                "The mysterious voice returns, now sounding amused. \"Congratulations on solving the first puzzle,\" it says. \"But did you really think escape would be so simple? You and your friend are now trapped in entangled Bloch spheres. Every action one takes affects the other.\"",

                "You try to move toward Sonalika, but the more you struggle to approach, the further they seem to drift away. When you rotate clockwise, they rotate counterclockwise. When you try to shift your phase, theirs shifts in complementary patterns.",

                "\"The only way out,\" the voice continues, \"is to master the gates of entanglement. The PSWAP to exchange your positions, the CNOT to flip states conditionally, and the CR for controlled rotations. Only by working together can you break free of this quantum prison.\"",

                "Sonalika looks at you with determination in their eyes. \"We can figure this out,\" they say. \"But we'll need to coordinate our actions perfectly. When I apply my gate, you'll need to apply yours at exactly the right moment.\"",

                "You notice three new controls have appeared on your quantum interface: PSWAP, CNOT, and CR, each with parameters that need precise calibration. A wrong move could entangle you both more deeply, perhaps irreversibly.",

                "\"Ready to try?\" Sonalika asks, hovering a finger over their control panel. The true test has only just begun."
            ]

        # Initialize state if not already done
        rho_2qubit_initial = two_qubit_swap_state(0.2, 0.3, np.pi / 4, 0.1)

        if 'level4_initial_state_rho' not in st.session_state:
            st.session_state.level4_initial_state_rho = rho_2qubit_initial
            st.session_state.level4_initial_state_rho1 = rho_1(rho_2qubit_initial)
            st.session_state.level4_initial_state_rho2 = rho_2(rho_2qubit_initial)
            st.session_state.level4_state_rho = rho_2qubit_initial
            st.session_state.level4_state_rho1 = rho_1(rho_2qubit_initial)
            st.session_state.level4_state_rho2 = rho_2(rho_2qubit_initial)

            # Initialize combined gate history - each entry is a tuple (time_step, qubit, gate_name)
            st.session_state.level4_combined_gate_history = []
            st.session_state.level4_time_step = 0

            # Define target state
            not_gate = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, 1],
                                 [0, 0, 1, 0]])  # CNOT

            # Function for controlled rotation gate
            cr_gate_func = lambda theta: np.array([[1, 0, 0, 0],
                                                   [0, 1, 0, 0],
                                                   [0, 0, 1, 0],
                                                   [0, 0, 0, np.exp(1j * theta)]])

            # Function for parameterized swap gate
            swap_gate_func = lambda theta: np.array([
                [1, 0, 0, 0],
                [0, np.cos(theta), 1j * np.sin(theta), 0],
                [0, 1j * np.sin(theta), np.cos(theta), 0],
                [0, 0, 0, 1]
            ])

            # Calculate target state by applying gates
            state_after_not = np.dot(np.dot(not_gate, rho_2qubit_initial), not_gate.T)
            cr_with_theta = cr_gate_func(3)
            state_after_cr = np.dot(np.dot(cr_with_theta, state_after_not), cr_with_theta.T.conj())
            swap_with_theta = swap_gate_func(4.06)
            final_state = np.dot(np.dot(swap_with_theta, state_after_cr), swap_with_theta.T.conj())

            st.session_state.level4_final_state_rho = final_state
            st.session_state.level4_final_state_rho1 = rho_1(final_state)
            st.session_state.level4_final_state_rho2 = rho_2(final_state)
            st.session_state.level4_final_bloch_vector1 = bloch_vector(st.session_state.level4_final_state_rho1)
            st.session_state.level4_final_bloch_vector2 = bloch_vector(st.session_state.level4_final_state_rho2)

        # Add custom CSS for story display
        st.markdown("""
        <style>
        .story-container {
            background-color: rgba(20, 20, 60, 0.8);
            color: #f0e6ff;
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0;
            border: 2px solid #4d79ff;
            font-family: 'Share Tech Mono', monospace;
            box-shadow: 0 0 15px rgba(77, 121, 255, 0.5);
        }
        .gate-info {
            background-color: rgba(40, 10, 70, 0.7);
            color: #f0e6ff;
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
            border: 2px solid #9966ff;
            font-family: 'Share Tech Mono', monospace;
            box-shadow: 0 0 15px rgba(153, 102, 255, 0.5);
        }
        .translucent-container {
            background-color: rgba(240, 240, 255, 0.2);
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        </style>
        """, unsafe_allow_html=True)

        # Content area
        if st.session_state.level4_story_index < len(st.session_state.level4_story_content):
            # Show story
            st.markdown(f"""
            <div class="story-container">
                <p style="font-size: 1.1em; line-height: 1.5;">{st.session_state.level4_story_content[st.session_state.level4_story_index]}</p>
            </div>
            """, unsafe_allow_html=True)

            # Continue button
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("Continue", key="level4_story_next"):
                    st.session_state.level4_story_index += 1
                    st.rerun()
        else:
            # After story is complete, show the entangled qubits interface

            # Display gate information
            st.markdown("""
            <div class="gate-info">
                <h3>Entanglement Gates</h3>
                <p>Use these special gates to manipulate the entangled qubits:</p>
            </div>
            """, unsafe_allow_html=True)

            # Display the mathematical forms of the gates
            st.latex(r'''
            \text{CNOT} = 
            \begin{pmatrix} 
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 1 \\
            0 & 0 & 1 & 0
            \end{pmatrix}
            \quad
            \text{PSWAP}(\phi) = 
            \begin{pmatrix} 
            1 & 0 & 0 & 0 \\
            0 & 0 & e^{i\phi} & 0 \\
            0 & e^{i\phi} & 0 & 0 \\
            0 & 0 & 0 & 1
            \end{pmatrix}
            \quad
            \text{CR}(\theta) = 
            \begin{pmatrix} 
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & 1 & 0 \\
            0 & 0 & 0 & e^{i\theta}
            \end{pmatrix}
            ''')

            # Create layout for the two qubits
            col1, col2, col3, col4 = st.columns([1, 5, 5, 1])

            with col2:
                st.header("Your Qubit")

                # Create placeholders for visualizations
                bloch_placeholder1 = st.empty()
                rho_placeholder1 = st.empty()

                # Display Bloch sphere for first qubit
                current_bloch_vector1 = bloch_vector(st.session_state.level4_state_rho1)
                with bloch_placeholder1:
                    plot_bloch_sphere(current_bloch_vector1, st.session_state.level4_final_bloch_vector1, 'blues',
                                      "Your Quantum State")

                # Gates for the first qubit
                st.write("### Apply Gates:")
                st.markdown('<div class="translucent-container">', unsafe_allow_html=True)

                # Single qubit controls
                subcol1, subcol2 = st.columns(2)
                with subcol1:
                    x_button1 = st.button('X Gate', key="x_gate_1")
                    y_button1 = st.button('Y Gate', key="y_gate_1")
                    z_button1 = st.button('Z Gate', key="z_gate_1")

                with subcol2:
                    # Two-qubit gates
                    cnot_button1 = st.button('CNOT', key="cnot_gate_1")

                    pswap_theta1 = st.slider('PSWAP φ (radians)',
                                             min_value=0.0, max_value=2 * np.pi,
                                             value=np.pi / 2, step=0.01, key="pswap_slider_1")
                    pswap_button1 = st.button('PSWAP', key="pswap_gate_1")

                    cr_theta1 = st.slider('CR θ (radians)',
                                          min_value=0.0, max_value=2 * np.pi,
                                          value=np.pi / 4, step=0.01, key="cr_slider_1")
                    cr_button1 = st.button('CR', key="cr_gate_1")

                st.markdown('</div>', unsafe_allow_html=True)

                # Display density matrix
                with rho_placeholder1:
                    display_rho(st.session_state.level4_state_rho1)

            with col3:
                st.header("Sonalika's Qubit")

                # Create placeholders for visualizations
                bloch_placeholder2 = st.empty()
                rho_placeholder2 = st.empty()

                # Display Bloch sphere for second qubit
                current_bloch_vector2 = bloch_vector(st.session_state.level4_state_rho2)
                with bloch_placeholder2:
                    plot_bloch_sphere(current_bloch_vector2, st.session_state.level4_final_bloch_vector2, 'purples',
                                      "Sonalika's Quantum State")

                # Gates for the second qubit
                st.write("### Apply Gates:")
                st.markdown('<div class="translucent-container">', unsafe_allow_html=True)

                # Single qubit controls
                subcol1, subcol2 = st.columns(2)
                with subcol1:
                    x_button2 = st.button('X Gate', key="x_gate_2")
                    y_button2 = st.button('Y Gate', key="y_gate_2")
                    z_button2 = st.button('Z Gate', key="z_gate_2")

                with subcol2:
                    # Two-qubit gates
                    cnot_button2 = st.button('CNOT', key="cnot_gate_2")

                    pswap_theta2 = st.slider('PSWAP φ (radians)',
                                             min_value=0.0, max_value=2 * np.pi,
                                             value=np.pi / 2, step=0.01, key="pswap_slider_2")
                    pswap_button2 = st.button('PSWAP', key="pswap_gate_2")

                    cr_theta2 = st.slider('CR θ (radians)',
                                          min_value=0.0, max_value=2 * np.pi,
                                          value=np.pi / 4, step=0.01, key="cr_slider_2")
                    cr_button2 = st.button('CR', key="cr_gate_2")

                st.markdown('</div>', unsafe_allow_html=True)

                # Display density matrix
                with rho_placeholder2:
                    display_rho(st.session_state.level4_state_rho2)

            # Extract timeline-based gate histories for visualization
            gate_history_q1 = []
            gate_history_q2 = []

            if len(st.session_state.level4_combined_gate_history) > 0:
                # Sort by time step to ensure proper order
                sorted_history = sorted(st.session_state.level4_combined_gate_history, key=lambda x: x[0])

                # Get the maximum time step
                max_time = sorted_history[-1][0] + 1

                # Initialize empty histories with None placeholders for each time step
                gate_history_q1 = [None] * max_time
                gate_history_q2 = [None] * max_time

                # Fill in gates at appropriate time steps
                for time, qubit, gate in sorted_history:
                    if qubit == 1:
                        gate_history_q1[time] = gate
                    elif qubit == 2:
                        gate_history_q2[time] = gate
                    elif qubit == "both":  # For two-qubit gates that affect both
                        gate_history_q1[time] = gate
                        gate_history_q2[time] = gate

                # Remove None values
                gate_history_q1 = [g for g in gate_history_q1 if g is not None]
                gate_history_q2 = [g for g in gate_history_q2 if g is not None]

            # Process button clicks for Qubit 1
            if x_button1:
                gate = pauli_x()
                st.session_state.level4_state_rho = np.kron(gate,
                                                            np.eye(2)) @ st.session_state.level4_state_rho @ np.kron(
                    gate.conj().T, np.eye(2))
                # Record gate in the combined history with time step
                st.session_state.level4_combined_gate_history.append((st.session_state.level4_time_step, 1, 'X'))
                st.session_state.level4_time_step += 1
                st.session_state.level4_state_rho1 = rho_1(st.session_state.level4_state_rho)
                st.session_state.level4_state_rho2 = rho_2(st.session_state.level4_state_rho)
                st.rerun()

            elif y_button1:
                gate = pauli_y()
                st.session_state.level4_state_rho = np.kron(gate,
                                                            np.eye(2)) @ st.session_state.level4_state_rho @ np.kron(
                    gate.conj().T, np.eye(2))
                # Record gate in the combined history with time step
                st.session_state.level4_combined_gate_history.append((st.session_state.level4_time_step, 1, 'Y'))
                st.session_state.level4_time_step += 1
                st.session_state.level4_state_rho1 = rho_1(st.session_state.level4_state_rho)
                st.session_state.level4_state_rho2 = rho_2(st.session_state.level4_state_rho)
                st.rerun()

            elif z_button1:
                gate = pauli_z()
                st.session_state.level4_state_rho = np.kron(gate,
                                                            np.eye(2)) @ st.session_state.level4_state_rho @ np.kron(
                    gate.conj().T, np.eye(2))
                # Record gate in the combined history with time step
                st.session_state.level4_combined_gate_history.append((st.session_state.level4_time_step, 1, 'Z'))
                st.session_state.level4_time_step += 1
                st.session_state.level4_state_rho1 = rho_1(st.session_state.level4_state_rho)
                st.session_state.level4_state_rho2 = rho_2(st.session_state.level4_state_rho)
                st.rerun()

            elif cnot_button1:
                gate = cnot_gate()
                st.session_state.level4_state_rho = np.dot(np.dot(gate, st.session_state.level4_state_rho), gate.T)
                # Record gate in the combined history with time step (affects both qubits)
                st.session_state.level4_combined_gate_history.append(
                    (st.session_state.level4_time_step, "both", 'CNOT'))
                st.session_state.level4_time_step += 1
                st.session_state.level4_state_rho1 = rho_1(st.session_state.level4_state_rho)
                st.session_state.level4_state_rho2 = rho_2(st.session_state.level4_state_rho)
                st.rerun()

            elif pswap_button1:
                gate = pswap_gate(pswap_theta1)
                st.session_state.level4_state_rho = np.dot(np.dot(gate, st.session_state.level4_state_rho),
                                                           gate.conj().T)
                # Record gate in the combined history with time step (affects both qubits)
                st.session_state.level4_combined_gate_history.append(
                    (st.session_state.level4_time_step, "both", f'PSWAP({pswap_theta1:.2f})'))
                st.session_state.level4_time_step += 1
                st.session_state.level4_state_rho1 = rho_1(st.session_state.level4_state_rho)
                st.session_state.level4_state_rho2 = rho_2(st.session_state.level4_state_rho)
                st.rerun()

            elif cr_button1:
                gate = cr_gate(cr_theta1)
                st.session_state.level4_state_rho = gate @ st.session_state.level4_state_rho @ gate.conj().T
                # Record gate in the combined history with time step (affects both qubits)
                st.session_state.level4_combined_gate_history.append(
                    (st.session_state.level4_time_step, "both", f'CR({cr_theta1:.2f})'))
                st.session_state.level4_time_step += 1
                st.session_state.level4_state_rho1 = rho_1(st.session_state.level4_state_rho)
                st.session_state.level4_state_rho2 = rho_2(st.session_state.level4_state_rho)
                st.rerun()

            # Process button clicks for Qubit 2
            elif x_button2:
                gate = pauli_x()
                st.session_state.level4_state_rho = np.kron(np.eye(2),
                                                            gate) @ st.session_state.level4_state_rho @ np.kron(
                    np.eye(2), gate.conj().T)
                # Record gate in the combined history with time step
                st.session_state.level4_combined_gate_history.append((st.session_state.level4_time_step, 2, 'X'))
                st.session_state.level4_time_step += 1
                st.session_state.level4_state_rho1 = rho_1(st.session_state.level4_state_rho)
                st.session_state.level4_state_rho2 = rho_2(st.session_state.level4_state_rho)
                st.rerun()

            elif y_button2:
                gate = pauli_y()
                st.session_state.level4_state_rho = np.kron(np.eye(2),
                                                            gate) @ st.session_state.level4_state_rho @ np.kron(
                    np.eye(2), gate.conj().T)
                # Record gate in the combined history with time step
                st.session_state.level4_combined_gate_history.append((st.session_state.level4_time_step, 2, 'Y'))
                st.session_state.level4_time_step += 1
                st.session_state.level4_state_rho1 = rho_1(st.session_state.level4_state_rho)
                st.session_state.level4_state_rho2 = rho_2(st.session_state.level4_state_rho)
                st.rerun()

            elif z_button2:
                gate = pauli_z()
                st.session_state.level4_state_rho = np.kron(np.eye(2),
                                                            gate) @ st.session_state.level4_state_rho @ np.kron(
                    np.eye(2), gate.conj().T)
                # Record gate in the combined history with time step
                st.session_state.level4_combined_gate_history.append((st.session_state.level4_time_step, 2, 'Z'))
                st.session_state.level4_time_step += 1
                st.session_state.level4_state_rho1 = rho_1(st.session_state.level4_state_rho)
                st.session_state.level4_state_rho2 = rho_2(st.session_state.level4_state_rho)
                st.rerun()

            elif cnot_button2:
                gate = cnot_gate()
                st.session_state.level4_state_rho = np.dot(np.dot(gate, st.session_state.level4_state_rho), gate.T)
                # Record gate in the combined history with time step (affects both qubits)
                st.session_state.level4_combined_gate_history.append(
                    (st.session_state.level4_time_step, "both", 'CNOT'))
                st.session_state.level4_time_step += 1
                st.session_state.level4_state_rho1 = rho_1(st.session_state.level4_state_rho)
                st.session_state.level4_state_rho2 = rho_2(st.session_state.level4_state_rho)
                st.rerun()

            elif pswap_button2:
                gate = pswap_gate(pswap_theta2)
                st.session_state.level4_state_rho = np.dot(np.dot(gate, st.session_state.level4_state_rho),
                                                           gate.conj().T)
                # Record gate in the combined history with time step (affects both qubits)
                st.session_state.level4_combined_gate_history.append(
                    (st.session_state.level4_time_step, "both", f'PSWAP({pswap_theta2:.2f})'))
                st.session_state.level4_time_step += 1
                st.session_state.level4_state_rho1 = rho_1(st.session_state.level4_state_rho)
                st.session_state.level4_state_rho2 = rho_2(st.session_state.level4_state_rho)
                st.rerun()


            elif cr_button2:

                gate = cr_gate(cr_theta2)

                st.session_state.level4_state_rho = gate @ st.session_state.level4_state_rho @ gate.conj().T

                # Record gate in the combined history with time step (affects both qubits)

                st.session_state.level4_combined_gate_history.append(
                    (st.session_state.level4_time_step, "both", f'CR({cr_theta2:.2f})'))

                st.session_state.level4_time_step += 1

                st.session_state.level4_state_rho1 = rho_1(st.session_state.level4_state_rho)

                st.session_state.level4_state_rho2 = rho_2(st.session_state.level4_state_rho)

                st.rerun()

                # Display circuit history

            st.markdown("### Two-Qubit Circuit History")

            # Create accurate circuit visualization based on the combined history

            max_time = st.session_state.level4_time_step

            if max_time > 0:

                # Sort history by time

                sorted_history = sorted(st.session_state.level4_combined_gate_history, key=lambda x: x[0])

                # Create the circuit figure

                circuit_fig = go.Figure()

                # Add qubit wires (length based on the number of time steps)

                circuit_fig.add_trace(go.Scatter(

                    x=[0, max_time],

                    y=[1, 1],

                    mode="lines",

                    line=dict(width=3, color="black"),

                    name="Your Qubit"

                ))

                circuit_fig.add_trace(go.Scatter(

                    x=[0, max_time],

                    y=[0, 0],

                    mode="lines",

                    line=dict(width=3, color="black"),

                    name="Sonalika's Qubit"

                ))

                # Add labels for the qubits

                circuit_fig.add_annotation(

                    x=-0.5, y=1,

                    text="Your Qubit",

                    showarrow=False,

                    font=dict(size=14, color="black")

                )

                circuit_fig.add_annotation(

                    x=-0.5, y=0,

                    text="Sonalika",

                    showarrow=False,

                    font=dict(size=14, color="black")

                )

                # Add gates at appropriate positions

                for time_step, qubit, gate_name in sorted_history:

                    if qubit == 1 or qubit == "both":
                        # Add gate for qubit 1

                        circuit_fig.add_trace(go.Scatter(

                            x=[time_step], y=[1],

                            mode="markers+text",

                            marker=dict(symbol="square", size=70, color="pink"),

                            text=[gate_name],

                            textposition='middle center',
                            textfont=dict(size=25, color="black"),

                            name=f"Gate at t={time_step}"

                        ))

                    if qubit == 2 or qubit == "both":
                        # Add gate for qubit 2

                        circuit_fig.add_trace(go.Scatter(

                            x=[time_step], y=[0],

                            mode="markers+text",

                            marker=dict(symbol="square", size=70, color="skyblue"),

                            text=[gate_name],

                            textposition='middle center',
                            textfont=dict(size=25, color="black"),

                            name=f"Gate at t={time_step}"

                        ))

                    # If it's a two-qubit gate, draw a line connecting the qubits

                    if qubit == "both":
                        circuit_fig.add_trace(go.Scatter(

                            x=[time_step, time_step],

                            y=[0, 1],

                            mode="lines",

                            line=dict(width=2, color="purple", dash="dot"),

                            name=f"Connection at t={time_step}"

                        ))

                # Update layout

                circuit_fig.update_layout(

                    xaxis=dict(range=[-1, max_time + 0.5], zeroline=False, showticklabels=True, title="Time Step"),

                    yaxis=dict(range=[-0.5, 1.5], zeroline=False, showticklabels=False),

                    margin=dict(l=30, r=30, b=30, t=30),

                    height=250,

                    width=700,

                    showlegend=False,

                    title="Two-Qubit Circuit History"

                )

                # Display the circuit

                st.plotly_chart(circuit_fig, key=f"two_qubit_circuit_{max_time}")

                # Show gate history as text for reference

                st.write("Gate sequence:")

                for time_step, qubit, gate_name in sorted_history:

                    if qubit == "both":

                        qubit_text = "Both qubits"

                    elif qubit == 1:

                        qubit_text = "Your qubit"

                    else:

                        qubit_text = "Sonalika's qubit"

                    st.write(f"Step {time_step}: {gate_name} on {qubit_text}")

            else:

                st.write("No gates applied yet. Apply gates to see the circuit history.")

            # Check if target state is reached (approx)

            target_vector1 = st.session_state.level4_final_bloch_vector1

            target_vector2 = st.session_state.level4_final_bloch_vector2

            current_vector1 = bloch_vector(st.session_state.level4_state_rho1)

            current_vector2 = bloch_vector(st.session_state.level4_state_rho2)

            distance1 = np.linalg.norm(target_vector1 - current_vector1)

            distance2 = np.linalg.norm(target_vector2 - current_vector2)

            # Level completion

            if distance1 < 0.1 and distance2 < 0.1:
                st.balloons()

                st.success(
                    "You've successfully coordinated with Sonalika to reach the target state! The entangled Bloch spheres begin to separate...")

            # Show how close they are to solution

            st.progress(max(0, 1.0 - (distance1 + distance2) / 2))

            score = 0

            if distance1 < 0.1 and distance2 < 0.1 or st.button("Click when done!"):
                score = 50

            if level(score):
                level_transition()

            st.button("Take me to next level")

    elif st.session_state.current_level == 5:
        st.header("Level 5: Quantum Rotation Masters")

        # Define the rho functions for reduced density matrices
        def rho_1(rho_2qubit):
            return np.array([[rho_2qubit[0, 0] + rho_2qubit[1, 1], rho_2qubit[0, 2] + rho_2qubit[1, 3]],
                             [rho_2qubit[2, 0] + rho_2qubit[3, 1], rho_2qubit[2, 2] + rho_2qubit[3, 3]]])

        def rho_2(rho_2qubit):
            return np.array([[rho_2qubit[0, 0] + rho_2qubit[2, 2], rho_2qubit[0, 1] + rho_2qubit[2, 3]],
                             [rho_2qubit[1, 0] + rho_2qubit[3, 2], rho_2qubit[1, 1] + rho_2qubit[3, 3]]])

        # Story initialization
        if 'level5_story_index' not in st.session_state:
            st.session_state.level5_story_index = 0
            st.session_state.level5_story_content = [
                "You input the final sequence of gates, and the door on the Bloch sphere slides open with a resonant hum. Relief washes over your quantum state as you rush through the opening.",

                "But something isn't right. As your consciousness settles, you realize you're still in a Bloch sphere—just a different one. And there, across the curved quantum landscape, is Sonalika, your friend who disappeared a week before you did. Their quantum signature flickers with recognition when they see you.",

                "\"Impressive progress,\" it says, \"but did you really think escape would be so simple? You and your friend are now trapped in entangled Bloch spheres.... \"In this realm, the single qubit gates are too simplistic. Now, you must harness the power of entanglement and multi-qubit gates.\"",

                "\"You made it!\" Sonalika calls out, their voice carrying across the quantum void. \"I've been waiting for—\" They stop mid-sentence as both of you suddenly lurch sideways. When you shifted your position, Sonalika moved too—perfectly mirroring your rotation but in the opposite direction.",

                "You try to move toward Sonalika, but the more you struggle to approach, the further they seem to drift away. When you rotate clockwise, they rotate counterclockwise. When you try to shift your phase, theirs shifts in complementary patterns.",

                "\"The only way out,\" the voice continues, \"is to master the gates of entanglement. The PSWAP to exchange your positions, the CNOT to flip states conditionally, and the CR for controlled rotations. Only by working together can you break free of this quantum prison.\"",

                "Sonalika looks at you with determination in their eyes. \"We can figure this out,\" they say. After all, rotation gates are universal—they can create any single-qubit transformation with the right sequence and parameters. But we'll need to coordinate our actions perfectly. When I apply my gate, you'll need to apply yours at exactly the right moment.\"",

                "You notice three new controls have appeared on your quantum interface: PSWAP, CNOT, and CR, each with parameters that need precise calibration. A wrong move could entangle you both more deeply, perhaps irreversibly.",

                "\"Ready to try?\" Sonalika asks, hovering a finger over their control panel. The true test has only just begun."
                
                "The challenge is clear: use rotation gates with the right parameters, along with two-qubit entangling operations, to reach the target state that will finally unlock your escape from this quantum puzzle."
            ]

        def cnot_gate_q1_controls_q2():
            # CNOT where qubit 1 controls qubit 2
            return np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 1],
                             [0, 0, 1, 0]])

        def cnot_gate_q2_controls_q1():
            # CNOT where qubit 2 controls qubit 1
            return np.array([[1, 0, 0, 0],
                             [0, 0, 0, 1],
                             [0, 0, 1, 0],
                             [0, 1, 0, 0]])

        def cr_gate_q1_controls_q2(theta):
            # CR where qubit 1 controls qubit 2
            return np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, np.exp(1j * theta)]])

        def cr_gate_q2_controls_q1(theta):
            # CR where qubit 2 controls qubit 1
            return np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, np.exp(1j * theta), 0],
                             [0, 0, 0, 1]])

        # Initialize state if not already done
        rho_2qubit_initial = two_qubit_swap_state(0.25, 0.35, np.pi / 4, 0.15)

        if 'level5_initial_state_rho' not in st.session_state:
            st.session_state.level5_initial_state_rho = rho_2qubit_initial
            st.session_state.level5_initial_state_rho1 = rho_1(rho_2qubit_initial)
            st.session_state.level5_initial_state_rho2 = rho_2(rho_2qubit_initial)
            st.session_state.level5_state_rho = rho_2qubit_initial
            st.session_state.level5_state_rho1 = rho_1(rho_2qubit_initial)
            st.session_state.level5_state_rho2 = rho_2(rho_2qubit_initial)

            # Initialize combined gate history - each entry is a tuple (time_step, qubit, gate_name)
            st.session_state.level5_combined_gate_history = []
            st.session_state.level5_time_step = 0

            # Define target state
            # Add these functions to handle different control directions


            # Function for parameterized swap gate
            swap_gate_func = lambda theta: np.array([
                [1, 0, 0, 0],
                [0, np.cos(theta), 1j * np.sin(theta), 0],
                [0, 1j * np.sin(theta), np.cos(theta), 0],
                [0, 0, 0, 1]
            ])

            # Calculate target state by applying gates
            # Update the target state generation
            state_after_not = np.dot(np.dot(cnot_gate_q1_controls_q2(), rho_2qubit_initial),
                                     cnot_gate_q1_controls_q2().T)
            cr_with_theta = cr_gate_q2_controls_q1(3.5)
            state_after_cr = np.dot(np.dot(cr_with_theta, state_after_not), cr_with_theta.T.conj())
            swap_with_theta = swap_gate_func(4.2)
            final_state = np.dot(np.dot(swap_with_theta, state_after_cr), swap_with_theta.T.conj())

            st.session_state.level5_final_state_rho = final_state
            st.session_state.level5_final_state_rho1 = rho_1(final_state)
            st.session_state.level5_final_state_rho2 = rho_2(final_state)
            st.session_state.level5_final_bloch_vector1 = bloch_vector(st.session_state.level5_final_state_rho1)
            st.session_state.level5_final_bloch_vector2 = bloch_vector(st.session_state.level5_final_state_rho2)

        # Add custom CSS for story display
        st.markdown("""
        <style>
        .story-container {
            background-color: rgba(20, 20, 60, 0.8);
            color: #f0e6ff;
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0;
            border: 2px solid #4d79ff;
            font-family: 'Share Tech Mono', monospace;
            box-shadow: 0 0 15px rgba(77, 121, 255, 0.5);
        }
        .gate-info {
            background-color: rgba(20, 10, 70, 0.7);
            color: #f0e6ff;
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
            border: 2px solid #9966ff;
            font-family: 'Share Tech Mono', monospace;
            box-shadow: 0 0 15px rgba(153, 102, 255, 0.5);
        }
        .translucent-container {
            background-color: rgba(240, 240, 255, 0.2);
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        </style>
        """, unsafe_allow_html=True)

        # Content area
        if st.session_state.level5_story_index < len(st.session_state.level5_story_content):
            # Show story
            st.markdown(f"""
            <div class="story-container">
                <p style="font-size: 1.1em; line-height: 1.5;">{st.session_state.level5_story_content[st.session_state.level5_story_index]}</p>
            </div>
            """, unsafe_allow_html=True)

            st.latex(r'''
                        R_y(\theta) = 
                        \begin{pmatrix} 
                        \cos(\theta/2) & -\sin(\theta/2) \\
                        \sin(\theta/2) & \cos(\theta/2)
                        \end{pmatrix}
                        \quad
                        R_z(\phi) = 
                        \begin{pmatrix} 
                        e^{-i\phi/2} & 0 \\
                        0 & e^{i\phi/2}
                        \end{pmatrix}
                        \quad
                        \text{Phase}(\delta) = 
                        \begin{pmatrix} 
                        1 & 0 \\
                        0 & e^{i\delta}
                        \end{pmatrix}
                        ''')

            st.latex(r'''
                        \text{CNOT} = 
                        \begin{pmatrix} 
                        1 & 0 & 0 & 0 \\
                        0 & 1 & 0 & 0 \\
                        0 & 0 & 0 & 1 \\
                        0 & 0 & 1 & 0
                        \end{pmatrix}
                        \quad
                        \text{PSWAP}(\phi) = 
                        \begin{pmatrix} 
                        1 & 0 & 0 & 0 \\
                        0 & \cos(\phi) & i\sin(\phi) & 0 \\
                        0 & i\sin(\phi) & \cos(\phi) & 0 \\
                        0 & 0 & 0 & 1
                        \end{pmatrix}
                        \quad
                        \text{CR}(\theta) = 
                        \begin{pmatrix} 
                        1 & 0 & 0 & 0 \\
                        0 & 1 & 0 & 0 \\
                        0 & 0 & 1 & 0 \\
                        0 & 0 & 0 & e^{i\theta}
                        \end{pmatrix}
                        ''')

            # Continue button
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("Continue", key="level5_story_next"):
                    st.session_state.level5_story_index += 1
                    st.rerun()
        else:
            # After story is complete, show the entangled qubits interface

            # Display gate information
            st.markdown("""
            <div class="gate-info">
                <h3>Rotation & Entanglement Gates</h3>
                <p>Use rotation gates for single-qubit operations and entangling gates for two-qubit operations:</p>
            </div>
            """, unsafe_allow_html=True)

            # Display the mathematical forms of the gates


            # Create layout for the two qubits
            col1, col2, col3, col4 = st.columns([1, 10, 10, 1])

            with col2:
                st.header("Your Qubit")

                # Create placeholders for visualizations
                bloch_placeholder1 = st.empty()
                rho_placeholder1 = st.empty()

                # Display Bloch sphere for first qubit
                current_bloch_vector1 = bloch_vector(st.session_state.level5_state_rho1)
                with bloch_placeholder1:
                    plot_bloch_sphere(current_bloch_vector1, st.session_state.level5_final_bloch_vector1, 'blues',
                                      "Your Quantum State")

                # Gates for the first qubit
                st.write("### Apply Gates:")
                st.markdown('<div class="translucent-container">', unsafe_allow_html=True)

                # Single qubit controls
                subcol1, subcol2 = st.columns(2)
                with subcol1:
                    # Y-Rotation Gate
                    ry_angle1 = st.slider('Ry angle (radians)',
                                          min_value=0.0, max_value=2 * np.pi,
                                          value=np.pi / 4, step=0.01, key="ry_slider_1")
                    ry_button1 = st.button('Ry', key="ry_gate_1")

                    # Z-Rotation Gate
                    rz_angle1 = st.slider('Rz angle (radians)',
                                          min_value=0.0, max_value=2 * np.pi,
                                          value=np.pi / 4, step=0.01, key="rz_slider_1")
                    rz_button1 = st.button('Rz', key="rz_gate_1")

                    # Phase Shift Gate
                    phase_angle1 = st.slider('Phase angle (radians)',
                                             min_value=0.0, max_value=2 * np.pi,
                                             value=np.pi / 4, step=0.01, key="phase_slider_1")
                    phase_button1 = st.button('Phase shift', key="phase_gate_1")

                with subcol2:
                    # Two-qubit gates
                    cnot_button1 = st.button('CNOT', key="cnot_gate_1")

                    pswap_theta1 = st.slider('PSWAP φ (radians)',
                                             min_value=0.0, max_value=2 * np.pi,
                                             value=np.pi / 2, step=0.01, key="pswap_slider_1")
                    pswap_button1 = st.button('PSWAP', key="pswap_gate_1")

                    cr_theta1 = st.slider('CR θ (radians)',
                                          min_value=0.0, max_value=2 * np.pi,
                                          value=np.pi / 4, step=0.01, key="cr_slider_1")
                    cr_button1 = st.button('CR', key="cr_gate_1")

                st.markdown('</div>', unsafe_allow_html=True)

                # Display density matrix
                with rho_placeholder1:
                    display_rho(st.session_state.level5_state_rho1)

            with col3:
                st.header("Sonalika's Qubit")

                # Create placeholders for visualizations
                bloch_placeholder2 = st.empty()
                rho_placeholder2 = st.empty()

                # Display Bloch sphere for second qubit
                current_bloch_vector2 = bloch_vector(st.session_state.level5_state_rho2)
                with bloch_placeholder2:
                    plot_bloch_sphere(current_bloch_vector2, st.session_state.level5_final_bloch_vector2, 'purples',
                                      "Sonalika's Quantum State")

                # Gates for the second qubit
                st.write("### Apply Gates:")
                st.markdown('<div class="translucent-container">', unsafe_allow_html=True)

                # Single qubit controls
                subcol1, subcol2 = st.columns(2)
                with subcol1:
                    # Y-Rotation Gate
                    ry_angle2 = st.slider('Ry angle (radians)',
                                          min_value=0.0, max_value=2 * np.pi,
                                          value=np.pi / 4, step=0.01,key="ry_slider_2")
                    ry_button2 = st.button('Ry', key="ry_gate_2")

                    # Z-Rotation Gate
                    rz_angle2 = st.slider('Rz angle (radians)',
                                          min_value=0.0, max_value=2 * np.pi,
                                          value=np.pi / 4, step=0.01, key="rz_slider_2")
                    rz_button2 = st.button('Rz', key="rz_gate_2")

                    # Phase Shift Gate
                    phase_angle2 = st.slider('Phase angle (radians)',
                                             min_value=0.0, max_value=2 * np.pi,
                                             value=np.pi / 4, step=0.01, key="phase_slider_2")
                    phase_button2 = st.button('Phase shift', key="phase_gate_2")

                with subcol2:
                    # Two-qubit gates
                    cnot_button2 = st.button('CNOT', key="cnot_gate_2")

                    pswap_theta2 = st.slider('PSWAP φ (radians)',
                                             min_value=0.0, max_value=2 * np.pi,
                                             value=np.pi / 2, step=0.01, key="pswap_slider_2")
                    pswap_button2 = st.button('PSWAP', key="pswap_gate_2")

                    cr_theta2 = st.slider('CR θ (radians)',
                                          min_value=0.0, max_value=2 * np.pi,
                                          value=np.pi / 4, step=0.01, key="cr_slider_2")
                    cr_button2 = st.button('CR', key="cr_gate_2")

                st.markdown('</div>', unsafe_allow_html=True)

                # Display density matrix
                with rho_placeholder2:
                    display_rho(st.session_state.level5_state_rho2)

            # Extract timeline-based gate histories for visualization
            gate_history_q1 = []
            gate_history_q2 = []

            if len(st.session_state.level5_combined_gate_history) > 0:
                # Sort by time step to ensure proper order
                sorted_history = sorted(st.session_state.level5_combined_gate_history, key=lambda x: x[0])

                # Get the maximum time step
                max_time = sorted_history[-1][0] + 1

                # Initialize empty histories with None placeholders for each time step
                gate_history_q1 = [None] * max_time
                gate_history_q2 = [None] * max_time

                # Fill in gates at appropriate time steps
                for time, qubit, gate in sorted_history:
                    if qubit == 1:
                        gate_history_q1[time] = gate
                    elif qubit == 2:
                        gate_history_q2[time] = gate
                    elif qubit == "both":  # For two-qubit gates that affect both
                        gate_history_q1[time] = gate
                        gate_history_q2[time] = gate

                # Remove None values
                gate_history_q1 = [g for g in gate_history_q1 if g is not None]
                gate_history_q2 = [g for g in gate_history_q2 if g is not None]

            # Process button clicks for Qubit 1
            if ry_button1:
                # Y-rotation gate
                gate = np.array([
                    [np.cos(ry_angle1 / 2), -np.sin(ry_angle1 / 2)],
                    [np.sin(ry_angle1 / 2), np.cos(ry_angle1 / 2)]
                ])
                st.session_state.level5_state_rho = np.kron(gate,
                                                            np.eye(2)) @ st.session_state.level5_state_rho @ np.kron(
                    gate.conj().T, np.eye(2))
                # Record gate in the combined history with time step
                st.session_state.level5_combined_gate_history.append(
                    (st.session_state.level5_time_step, 1, f'Ry({ry_angle1:.2f})'))
                st.session_state.level5_time_step += 1
                st.session_state.level5_state_rho1 = rho_1(st.session_state.level5_state_rho)
                st.session_state.level5_state_rho2 = rho_2(st.session_state.level5_state_rho)
                st.rerun()

            elif rz_button1:
                # Z-rotation gate
                gate = np.array([
                    [np.exp(-1j * rz_angle1 / 2), 0],
                    [0, np.exp(1j * rz_angle1 / 2)]
                ])
                st.session_state.level5_state_rho = np.kron(gate,
                                                            np.eye(2)) @ st.session_state.level5_state_rho @ np.kron(
                    gate.conj().T, np.eye(2))
                # Record gate in the combined history with time step
                st.session_state.level5_combined_gate_history.append(
                    (st.session_state.level5_time_step, 1, f'Rz({rz_angle1:.2f})'))
                st.session_state.level5_time_step += 1
                st.session_state.level5_state_rho1 = rho_1(st.session_state.level5_state_rho)
                st.session_state.level5_state_rho2 = rho_2(st.session_state.level5_state_rho)
                st.rerun()

            elif phase_button1:
                # Phase shift gate
                gate = np.array([
                    [1, 0],
                    [0, np.exp(1j * phase_angle1)]
                ])
                st.session_state.level5_state_rho = np.kron(gate,
                                                            np.eye(2)) @ st.session_state.level5_state_rho @ np.kron(
                    gate.conj().T, np.eye(2))
                # Record gate in the combined history with time step
                st.session_state.level5_combined_gate_history.append(
                    (st.session_state.level5_time_step, 1, f'Phase({phase_angle1:.2f})'))
                st.session_state.level5_time_step += 1
                st.session_state.level5_state_rho1 = rho_1(st.session_state.level5_state_rho)
                st.session_state.level5_state_rho2 = rho_2(st.session_state.level5_state_rho)
                st.rerun()


            elif cnot_button1:

                gate = cnot_gate_q1_controls_q2()  # Qubit 1 controls qubit 2

                st.session_state.level5_state_rho = np.dot(np.dot(gate, st.session_state.level5_state_rho), gate.T)

                # Record gate in the combined history with time step (affects both qubits)

                st.session_state.level5_combined_gate_history.append(

                    (st.session_state.level5_time_step, "both", 'CNOT (Q1→Q2)'))

                st.session_state.level5_time_step += 1

                st.session_state.level5_state_rho1 = rho_1(st.session_state.level5_state_rho)

                st.session_state.level5_state_rho2 = rho_2(st.session_state.level5_state_rho)

                st.rerun()

            elif pswap_button1:
                gate = pswap_gate(pswap_theta1)
                st.session_state.level5_state_rho = np.dot(np.dot(gate, st.session_state.level5_state_rho),
                                                           gate.conj().T)
                # Record gate in the combined history with time step (affects both qubits)
                st.session_state.level5_combined_gate_history.append(
                    (st.session_state.level5_time_step, "both", f'PSWAP({pswap_theta1:.2f})'))
                st.session_state.level5_time_step += 1
                st.session_state.level5_state_rho1 = rho_1(st.session_state.level5_state_rho)
                st.session_state.level5_state_rho2 = rho_2(st.session_state.level5_state_rho)
                st.rerun()


            elif cr_button1:

                gate = cr_gate_q1_controls_q2(cr_theta1)  # Qubit 1 controls qubit 2

                st.session_state.level5_state_rho = gate @ st.session_state.level5_state_rho @ gate.conj().T

                # Record gate in the combined history with time step (affects both qubits)

                st.session_state.level5_combined_gate_history.append(

                    (st.session_state.level5_time_step, "both", f'CR({cr_theta1:.2f}) (Q1→Q2)'))

                st.session_state.level5_time_step += 1

                st.session_state.level5_state_rho1 = rho_1(st.session_state.level5_state_rho)

                st.session_state.level5_state_rho2 = rho_2(st.session_state.level5_state_rho)

                st.rerun()

            # Process button clicks for Qubit 2
            elif ry_button2:
                # Y-rotation gate
                gate = np.array([
                    [np.cos(ry_angle2 / 2), -np.sin(ry_angle2 / 2)],
                    [np.sin(ry_angle2 / 2), np.cos(ry_angle2 / 2)]
                ])
                st.session_state.level5_state_rho = np.kron(np.eye(2),
                                                            gate) @ st.session_state.level5_state_rho @ np.kron(
                    np.eye(2), gate.conj().T)
                # Record gate in the combined history with time step
                st.session_state.level5_combined_gate_history.append(
                    (st.session_state.level5_time_step, 2, f'Ry({ry_angle2:.2f})'))
                st.session_state.level5_time_step += 1
                st.session_state.level5_state_rho1 = rho_1(st.session_state.level5_state_rho)
                st.session_state.level5_state_rho2 = rho_2(st.session_state.level5_state_rho)
                st.rerun()

            elif rz_button2:
                # Z-rotation gate
                gate = np.array([
                    [np.exp(-1j * rz_angle2 / 2), 0],
                    [0, np.exp(1j * rz_angle2 / 2)]
                ])
                st.session_state.level5_state_rho = np.kron(np.eye(2),
                                                            gate) @ st.session_state.level5_state_rho @ np.kron(
                    np.eye(2), gate.conj().T)
                # Record gate in the combined history with time step
                st.session_state.level5_combined_gate_history.append(
                    (st.session_state.level5_time_step, 2, f'Rz({rz_angle2:.2f})'))
                st.session_state.level5_time_step += 1
                st.session_state.level5_state_rho1 = rho_1(st.session_state.level5_state_rho)
                st.session_state.level5_state_rho2 = rho_2(st.session_state.level5_state_rho)
                st.rerun()

            elif phase_button2:
                # Phase shift gate
                gate = np.array([
                    [1, 0],
                    [0, np.exp(1j * phase_angle2)]
                ])
                st.session_state.level5_state_rho = np.kron(np.eye(2),
                                                            gate) @ st.session_state.level5_state_rho @ np.kron(
                    np.eye(2), gate.conj().T)
                # Record gate in the combined history with time step
                st.session_state.level5_combined_gate_history.append(
                    (st.session_state.level5_time_step, 2, f'Phase({phase_angle2:.2f})'))
                st.session_state.level5_time_step += 1
                st.session_state.level5_state_rho1 = rho_1(st.session_state.level5_state_rho)
                st.session_state.level5_state_rho2 = rho_2(st.session_state.level5_state_rho)
                st.rerun()


            elif cnot_button2:

                gate = cnot_gate_q2_controls_q1()  # Qubit 2 controls qubit 1

                st.session_state.level5_state_rho = np.dot(np.dot(gate, st.session_state.level5_state_rho), gate.T)

                # Record gate in the combined history with time step (affects both qubits)

                st.session_state.level5_combined_gate_history.append(

                    (st.session_state.level5_time_step, "both", 'CNOT (Q2→Q1)'))

                st.session_state.level5_time_step += 1

                st.session_state.level5_state_rho1 = rho_1(st.session_state.level5_state_rho)

                st.session_state.level5_state_rho2 = rho_2(st.session_state.level5_state_rho)

                st.rerun()

            elif pswap_button2:
                gate = pswap_gate(pswap_theta2)
                st.session_state.level5_state_rho = np.dot(np.dot(gate, st.session_state.level5_state_rho),
                                                           gate.conj().T)
                # Record gate in the combined history with time step (affects both qubits)
                st.session_state.level5_combined_gate_history.append(
                    (st.session_state.level5_time_step, "both", f'PSWAP({pswap_theta2:.2f})'))
                st.session_state.level5_time_step += 1
                st.session_state.level5_state_rho1 = rho_1(st.session_state.level5_state_rho)
                st.session_state.level5_state_rho2 = rho_2(st.session_state.level5_state_rho)
                st.rerun()


            elif cr_button2:

                gate = cr_gate_q2_controls_q1(cr_theta2)  # Qubit 2 controls qubit 1

                st.session_state.level5_state_rho = gate @ st.session_state.level5_state_rho @ gate.conj().T

                # Record gate in the combined history with time step (affects both qubits)

                st.session_state.level5_combined_gate_history.append(

                    (st.session_state.level5_time_step, "both", f'CR({cr_theta2:.2f}) (Q2→Q1)'))

                st.session_state.level5_time_step += 1

                st.session_state.level5_state_rho1 = rho_1(st.session_state.level5_state_rho)

                st.session_state.level5_state_rho2 = rho_2(st.session_state.level5_state_rho)

                st.rerun()

            # Display circuit history
            st.markdown("### Two-Qubit Circuit History")

            # Create accurate circuit visualization based on the combined history
            # Create accurate circuit visualization based on the combined history
            max_time = st.session_state.level5_time_step

            if max_time > 0:
                # Sort history by time
                sorted_history = sorted(st.session_state.level5_combined_gate_history, key=lambda x: x[0])

                # Create the circuit figure
                circuit_fig = go.Figure()

                # Add qubit wires (length based on the number of time steps)
                circuit_fig.add_trace(go.Scatter(
                    x=[0, max_time],
                    y=[1, 1],
                    mode="lines",
                    line=dict(width=3, color="black"),
                    name="Your Qubit"
                ))

                circuit_fig.add_trace(go.Scatter(
                    x=[0, max_time],
                    y=[0, 0],
                    mode="lines",
                    line=dict(width=3, color="black"),
                    name="Sonalika's Qubit"
                ))

                # Add labels for the qubits
                circuit_fig.add_annotation(
                    x=-0.5, y=1,
                    text="Your Qubit",
                    showarrow=False,
                    font=dict(size=14, color="black")
                )

                circuit_fig.add_annotation(
                    x=-0.5, y=0,
                    text="Sonalika",
                    showarrow=False,
                    font=dict(size=14, color="black")
                )

                # Add gates at appropriate positions
                for time_step, qubit, gate_name in sorted_history:
                    if qubit == 1 or qubit == "both":
                        # Add gate for qubit 1
                        circuit_fig.add_trace(go.Scatter(
                            x=[time_step], y=[1],
                            mode="markers+text",
                            marker=dict(symbol="square", size=70, color="pink"),
                            text=[gate_name],
                            textposition='middle center',
                            textfont=dict(size=25, color="black"),
                            #name=f"Gate at t={time_step}"
                        ))

                    if qubit == 2 or qubit == "both":
                        # Add gate for qubit 2
                        circuit_fig.add_trace(go.Scatter(
                            x=[time_step], y=[0],
                            mode="markers+text",
                            marker=dict(symbol="square", size=70, color="skyblue"),
                            text=[gate_name],
                            textposition='middle center',
                            textfont=dict(size=25, color="black"),
                            #name=f"Gate at t={time_step}"
                        ))

                    # If it's a two-qubit gate, draw a line connecting the qubits
                    if qubit == "both":
                        circuit_fig.add_trace(go.Scatter(
                            x=[time_step, time_step],
                            y=[0, 1],
                            mode="lines",
                            line=dict(width=2, color="purple", dash="dot"),
                            #name=f"Connection at t={time_step}"
                        ))

                # Update layout
                circuit_fig.update_layout(
                    xaxis=dict(range=[-1, max_time + 0.5], zeroline=False, showticklabels=True, title="Time Step"),
                    yaxis=dict(range=[-0.5, 1.5], zeroline=False, showticklabels=False),
                    margin=dict(l=30, r=30, b=30, t=30),
                    height=250,
                    width=700,
                    showlegend=False,
                    title="Two-Qubit Circuit History"
                )

                # Display the circuit
                st.plotly_chart(circuit_fig, key=f"two_qubit_circuit_{max_time}")

                # Show gate history as text for reference
                st.write("Gate sequence:")
                for time_step, qubit, gate_name in sorted_history:
                    if qubit == "both":
                        qubit_text = "Both qubits"
                    elif qubit == 1:
                        qubit_text = "Your qubit"
                    else:
                        qubit_text = "Sonalika's qubit"
                    #st.write(f"Step {time_step}: {gate_name} on {qubit_text}")

            else:
                st.write("No gates applied yet. Apply gates to see the circuit history.")

            # Check if target state is reached (approx)
            target_vector1 = st.session_state.level5_final_bloch_vector1
            target_vector2 = st.session_state.level5_final_bloch_vector2
            current_vector1 = bloch_vector(st.session_state.level5_state_rho1)
            current_vector2 = bloch_vector(st.session_state.level5_state_rho2)
            distance1 = np.linalg.norm(target_vector1 - current_vector1)
            distance2 = np.linalg.norm(target_vector2 - current_vector2)

            # Display progress bar for each qubit
            #st.subheader("Progress to Target States")

            col1, col2 = st.columns(2)
            with col1:
                st.write("Your qubit's progress:")
                progress1 = max(0, min(1.0, 1.0 - (distance1 / 2.0)))
                st.progress(progress1)
                st.write(f"Distance to target: {distance1:.6f}")

                if progress1 > 0.9:
                    st.success("Perfect! Your qubit is extremely close to the target state!")
                elif progress1 > 0.8:
                    st.info("Very close! Just a tiny adjustment needed for your qubit!")

            with col2:
                st.write("Sonalika's qubit's progress:")
                progress2 = max(0, min(1.0, 1.0 - (distance2 / 2.0)))
                st.progress(progress2)
                st.write(f"Distance to target: {distance2:.6f}")

                if progress2 > 0.9:
                    st.success("Perfect! Sonalika's qubit is extremely close to the target state!")
                elif progress2 > 0.8:
                    st.info("Very close! Just a tiny adjustment needed for Sonalika's qubit!")

            # Combined progress
            st.write("Overall progress:")
            combined_progress = max(0, min(1.0, 1.0 - (distance1 + distance2) / 4.0))
            st.progress(combined_progress)

            if combined_progress > 0.9:
                st.success("Amazing! Both qubits are aligned perfectly with their targets!")
            elif combined_progress > 0.85:
                st.info("You're very close to completing the challenge!")
            elif combined_progress > 0.75:
                st.info("Making good progress! Keep adjusting the gates.")
            elif combined_progress > 0.5:
                st.warning("You're on the right track, but still have some way to go.")
            else:
                st.error("Still far from the target states. Try different rotation angles and gate combinations.")

            # Level completion
            if distance1 < 0.1 and distance2 < 0.1:
                st.snow()
                st.success(
                    "You've successfully mastered the rotation gates and reached the target states! The quantum doors begin to glow with a bright light...")

                # Add a hint for the solution sequence
                with st.expander("Solution Sequence:"):
                    st.write("""
                            The target state can be reached with this sequence:
                            1. CNOT
                            2. CR(3.5) gate
                            3. PSWAP(4.2) gate

                            Combined with appropriate rotations on individual qubits:
                            - For your qubit: Ry(π/3) followed by Rz(π/4)
                            - For Sonalika's qubit: Phase(π/2) followed by Ry(π/6)
                            """)

            # Show how close they are to solution
            score = 0
            if distance1 < 0.1 and distance2 < 0.1 or st.button("Click when done!", key="level5_done"):
                score = 50
                if level(score):
                    level_transition()

            if st.button("Take me to next level", key="level5_next"):
                st.session_state.current_level = 7
                st.rerun()

    elif st.session_state.current_level == 6:
        st.header("Level 6: Deutsch's Algorithm - One Query Wonder!")

        st.markdown("""
        ### The Challenge
        You have a mystery function f(x). It's either:
        - **Constant**: f(0) = f(1) (both give same answer)
        - **Balanced**: f(0) ≠ f(1) (different answers)

        **Classical computer**: Needs 2 tries to be sure  
        **Quantum computer**: Needs only 1 try!
        """)

        # Initialize state
        if 'deutsch_step' not in st.session_state:
            st.session_state.deutsch_step = 0
            st.session_state.function_type = "constant"

        if st.session_state.deutsch_step == 0:
            # Step 1: Choose function
            st.markdown("### Step 1: Pick a mystery function to test")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Constant Function", use_container_width=True):
                    st.session_state.function_type = "constant"
                    st.session_state.deutsch_state = create_deutsch_density_matrix()
                    st.session_state.deutsch_step = 1
                    st.rerun()

            with col2:
                if st.button("Balanced Function", use_container_width=True):
                    st.session_state.function_type = "balanced"
                    st.session_state.deutsch_state = create_deutsch_density_matrix()
                    st.session_state.deutsch_step = 1
                    st.rerun()

        elif st.session_state.deutsch_step == 1:
            # Step 2: Initial setup
            st.markdown(f"Step 2: Start with |01⟩ - Testing {st.session_state.function_type} function")

            # Show initial Bloch spheres
            rho1 = extract_rho_1(st.session_state.deutsch_state)
            rho2 = extract_rho_2(st.session_state.deutsch_state)

            col1, col2 = st.columns(2)
            with col1:
                st.write("Input Qubit** (starts at |0⟩)")
                bloch1 = bloch_vector(rho1)
                plot_bloch_sphere(bloch1, np.array([0, 0, 0]), 'blues', "Input")

            with col2:
                st.write("Helper Qubit** (starts at |1⟩)")
                bloch2 = bloch_vector(rho2)
                plot_bloch_sphere(bloch2, np.array([0, 0, 0]), 'reds', "Helper")

            if st.button("Apply Hadamard → Create superposition"):
                # Apply Hadamard to first qubit
                st.session_state.deutsch_state = apply_hadamard_to_first_qubit(st.session_state.deutsch_state)
                st.session_state.deutsch_step = 2
                st.rerun()

        elif st.session_state.deutsch_step == 2:
            # Step 3: After Hadamard
            st.markdown("Step 3: Input qubit now tests BOTH 0 and 1 at once!")

            rho1 = extract_rho_1(st.session_state.deutsch_state)
            rho2 = extract_rho_2(st.session_state.deutsch_state)

            col1, col2 = st.columns(2)
            with col1:
                st.write("Input Qubit (superposition)")
                bloch1 = bloch_vector(rho1)
                plot_bloch_sphere(bloch1, np.array([0, 0, 0]), 'blues', "Superposition")

            with col2:
                st.write("Helper Qubit")
                bloch2 = bloch_vector(rho2)
                plot_bloch_sphere(bloch2, np.array([0, 0, 0]), 'reds', "Helper")

            if st.button(f"Query the {st.session_state.function_type} function"):
                # Apply oracle
                if st.session_state.function_type == "constant":
                    st.session_state.deutsch_state = oracle_constant(st.session_state.deutsch_state)
                else:
                    st.session_state.deutsch_state = oracle_balanced(st.session_state.deutsch_state)

                # Apply final Hadamard
                st.session_state.deutsch_state = apply_hadamard_to_first_qubit(st.session_state.deutsch_state)
                st.session_state.deutsch_step = 3
                st.rerun()

        elif st.session_state.deutsch_step == 3:
            # Step 4: Result
            st.markdown("Step 4: Read the answer!")

            rho1 = extract_rho_1(st.session_state.deutsch_state)
            bloch1 = bloch_vector(rho1)

            col1, col2 = st.columns(2)
            with col1:
                st.write("Final Input Qubit")
                plot_bloch_sphere(bloch1, np.array([0, 0, 0]), 'blues', "Result")

            with col2:
                # Determine result
                if bloch1[2] > 0.5:  # Close to |0⟩
                    result = "CONSTANT"
                    st.success("Result: CONSTANT function!")
                    st.write("Qubit is at |0⟩ → Function gives same output for both inputs")
                else:  # Close to |1⟩
                    result = "BALANCED"
                    st.success(" Result: BALANCED function!")
                    st.write("Qubit is at |1⟩ → Function gives different outputs")

                # Check if correct
                if result.lower() == st.session_state.function_type:
                    st.balloons()
                    st.success("Correct! Quantum wins!")
                else:
                    st.error("Oops, try again!")

            st.markdown("""
                The Magic:
            - Quantum superposition lets us test both inputs at once
            - One measurement tells us everything we need to know
            - Classical computers would need 2 separate tests!
            """)

            if st.button("Try another function"):
                st.session_state.deutsch_step = 0
                st.rerun()

            if st.button("Complete Level"):
                if level(50):
                    level_transition()

    # elif st.session_state.current_level == 1:
    #     st.header("Level 7: Quantum Teleportation - Beam Me Up!")
    #
    #     st.markdown("""
    #     ### The Mission
    #     Alice wants to send a quantum state to Bob instantly!
    #     But she can't just copy it (no-cloning theorem) or send it directly.
    #
    #     **Solution**: Use quantum entanglement + classical communication!
    #     """)
    #
    #     # Initialize state
    #     if 'teleport_step' not in st.session_state:
    #         st.session_state.teleport_step = 0
    #
    #     if st.session_state.teleport_step == 0:
    #         # Step 1: Setup
    #         st.markdown("### Step 1: The Setup")
    #         st.markdown("Alice has a qubit |+⟩ = (|0⟩ + |1⟩)/√2 to send to Bob")
    #
    #         # Alice's state to teleport
    #         alice_state = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
    #         alice_rho, _, _, _ = compute_pauli_expansion(alice_state)
    #
    #         # Bob starts with |0⟩
    #         bob_state = np.array([1, 0])
    #         bob_rho, _, _, _ = compute_pauli_expansion(bob_state)
    #
    #         st.session_state.alice_rho = alice_rho
    #         st.session_state.bob_rho = bob_rho
    #
    #         col1, col2 = st.columns(2)
    #         with col1:
    #             st.write("**Alice's Qubit** (to teleport)")
    #             alice_bloch = bloch_vector(alice_rho)
    #             plot_bloch_sphere(alice_bloch, np.array([0, 0, 0]), 'blues', "Alice's |+⟩")
    #             st.write("State: |+⟩ = (|0⟩ + |1⟩)/√2")
    #
    #         with col2:
    #             st.write("**Bob's Qubit** (destination)")
    #             bob_bloch = bloch_vector(bob_rho)
    #             plot_bloch_sphere(bob_bloch, np.array([0, 0, 0]), 'reds', "Bob's |0⟩")
    #             st.write("State: |0⟩")
    #
    #         if st.button("Create entanglement between Alice & Bob"):
    #             st.session_state.teleport_step = 1
    #             st.rerun()
    #
    #     elif st.session_state.teleport_step == 1:
    #         # Step 2: Entanglement
    #         st.markdown("### Step 2: Alice & Bob share entangled qubits")
    #         st.markdown("Now they both have maximally mixed states (center of Bloch sphere)")
    #
    #         # Both become maximally mixed due to entanglement
    #         mixed_state = 0.5 * np.eye(2)
    #         st.session_state.alice_aux = mixed_state
    #         st.session_state.bob_rho = mixed_state
    #
    #         col1, col2, col3 = st.columns(3)
    #         with col1:
    #             st.write("**Alice's original qubit**")
    #             alice_bloch = bloch_vector(st.session_state.alice_rho)
    #             plot_bloch_sphere(alice_bloch, np.array([0, 0, 0]), 'blues', "Still |+⟩")
    #
    #         with col2:
    #             st.write("**Alice's helper** (entangled)")
    #             aux_bloch = bloch_vector(st.session_state.alice_aux)
    #             plot_bloch_sphere(aux_bloch, np.array([0, 0, 0]), 'greens', "Entangled")
    #
    #         with col3:
    #             st.write("**Bob's qubit** (entangled)")
    #             bob_bloch = bloch_vector(st.session_state.bob_rho)
    #             plot_bloch_sphere(bob_bloch, np.array([0, 0, 0]), 'reds', "Entangled")
    #
    #         if st.button("Alice measures her qubits"):
    #             st.session_state.teleport_step = 2
    #             st.rerun()
    #
    #     elif st.session_state.teleport_step == 2:
    #         # Step 3: Measurement & Teleportation
    #         st.markdown("### Step 3: Measurement destroys Alice's qubits but transfers info to Bob!")
    #
    #         # Simulate measurement result
    #         measurement = np.random.choice(["00", "01", "10", "11"])
    #         st.session_state.measurement = measurement
    #
    #         st.info(f"Alice's measurement: {measurement}")
    #         st.info("Alice sends this result to Bob via classical communication 📞")
    #
    #         # Alice's qubits are destroyed (random states)
    #         destroyed_state = 0.5 * np.eye(2)
    #
    #         # Bob gets the teleported state (maybe with corrections needed)
    #         teleported_state = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
    #         bob_final_rho, _, _, _ = compute_pauli_expansion(teleported_state)
    #
    #         col1, col2, col3 = st.columns(3)
    #         with col1:
    #             st.write("**Alice's qubits** (destroyed)")
    #             destroyed_bloch = bloch_vector(destroyed_state)
    #             plot_bloch_sphere(destroyed_bloch, np.array([0, 0, 0]), 'Greys', "Destroyed")
    #             st.write("💥 Information gone!")
    #
    #         with col2:
    #             st.write("**Classical message**")
    #             st.markdown(f"### 📱 {measurement}")
    #             st.write("Sent to Bob")
    #
    #         with col3:
    #             st.write("**Bob's qubit** (teleported!)")
    #             bob_bloch = bloch_vector(bob_final_rho)
    #             plot_bloch_sphere(bob_bloch, np.array([0, 0, 0]), 'reds', "Teleported!")
    #             st.write("State: |+⟩ = (|0⟩ + |1⟩)/√2")
    #
    #         if st.button("Bob applies correction"):
    #             st.session_state.teleport_step = 3
    #             st.rerun()
    #
    #     elif st.session_state.teleport_step == 3:
    #         # Step 4: Success!
    #         st.markdown("### Step 4: Teleportation Complete! ✨")
    #
    #         corrections = {
    #             "00": "No correction needed",
    #             "01": "Apply X gate",
    #             "10": "Apply Z gate",
    #             "11": "Apply X and Z gates"
    #         }
    #
    #         st.success(f"Bob applies: {corrections[st.session_state.measurement]}")
    #
    #         # Show before and after
    #         col1, col2 = st.columns(2)
    #         with col1:
    #             st.write("**Original** (Alice had)")
    #             original_bloch = np.array([1, 0, 0])  # |+⟩ state
    #             plot_bloch_sphere(original_bloch, np.array([0, 0, 0]), 'blues', "Original |+⟩")
    #
    #         with col2:
    #             st.write("**Final** (Bob has)")
    #             final_bloch = np.array([1, 0, 0])  # |+⟩ state after correction
    #             plot_bloch_sphere(final_bloch, np.array([0, 0, 0]), 'reds', "Teleported |+⟩")
    #
    #         st.balloons()
    #         st.success("🎉 Perfect teleportation! The quantum state traveled instantly!")
    #
    #         st.markdown("""
    #         ### 🧠 Key Points:
    #         - ✅ Quantum information transferred instantly
    #         - ✅ Original qubit was destroyed (no cloning!)
    #         - ✅ Classical communication was needed
    #         - ✅ No faster-than-light communication
    #         """)
    #
    #         if st.button("🔄 Teleport again"):
    #             st.session_state.teleport_step = 0
    #             st.rerun()
    #
    #         if st.button("🎓 Complete Level"):
    #             if level(50):
    #                 level_transition()

    elif st.session_state.current_level == 9:

        st.header("Level 5: Maximally Entangled Spheres")

        # Define the rho functions for reduced density matrices
        def rho_1(rho_2qubit):
            return np.array([[rho_2qubit[0, 0] + rho_2qubit[1, 1], rho_2qubit[0, 2] + rho_2qubit[1, 3]],
                             [rho_2qubit[2, 0] + rho_2qubit[3, 1], rho_2qubit[2, 2] + rho_2qubit[3, 3]]])

        def rho_2(rho_2qubit):
            return np.array([[rho_2qubit[0, 0] + rho_2qubit[2, 2], rho_2qubit[0, 1] + rho_2qubit[2, 3]],
                             [rho_2qubit[1, 0] + rho_2qubit[3, 2], rho_2qubit[1, 1] + rho_2qubit[3, 3]]])

        # Story initialization
        if 'level5_story_index' not in st.session_state:
            st.session_state.level5_story_index = 0
            st.session_state.level5_story_content = [
                "You input the final sequence of gates, and the door on the Bloch sphere slides open with a resonant hum. Relief washes over your quantum state as you rush through the opening, feeling your wavefunction expand beyond the confines of your spherical prison.",

                "But something isn't right. As your consciousness settles, you realize you're still in a Bloch sphere—just a different one. And there, across the curved quantum landscape, is Sonalika, your friend who disappeared a week before you did. Their quantum signature flickers with recognition when they see you.",

                "\"You made it!\" Sonalika calls out, their voice carrying across the quantum void. \"I've been waiting for—\" They stop mid-sentence as both of you suddenly lurch sideways. When you shifted your position, Sonalika moved too—perfectly mirroring your rotation but in the opposite direction.",

                "The mysterious voice returns, now sounding amused. \"Congratulations on solving the first puzzle,\" it says. \"But did you really think escape would be so simple? You and your friend are now trapped in entangled Bloch spheres. Every action one takes affects the other.\"",

                "You try to move toward Sonalika, but the more you struggle to approach, the further they seem to drift away. When you rotate clockwise, they rotate counterclockwise. When you try to shift your phase, theirs shifts in complementary patterns.",

                "\"The only way out,\" the voice continues, \"is to master the gates of entanglement. The PSWAP to exchange your positions, the CNOT to flip states conditionally, and the CR for controlled rotations. Only by working together can you break free of this quantum prison.\"",

                "Sonalika looks at you with determination in their eyes. \"We can figure this out,\" they say. \"But we'll need to coordinate our actions perfectly. When I apply my gate, you'll need to apply yours at exactly the right moment.\"",

                "You notice three new controls have appeared on your quantum interface: PSWAP, CNOT, and CR, each with parameters that need precise calibration. A wrong move could entangle you both more deeply, perhaps irreversibly.",

                "\"Ready to try?\" Sonalika asks, hovering a finger over their control panel. The true test has only just begun."
            ]

        # Initialize state if not already done
        rho_2qubit_initial = bell_phi_minus

        if 'level5_initial_state_rho' not in st.session_state:
            st.session_state.level5_initial_state_rho = rho_2qubit_initial
            st.session_state.level5_initial_state_rho1 = rho_1(rho_2qubit_initial)
            st.session_state.level5_initial_state_rho2 = rho_2(rho_2qubit_initial)
            st.session_state.level5_state_rho = rho_2qubit_initial
            st.session_state.level5_state_rho1 = rho_1(rho_2qubit_initial)
            st.session_state.level5_state_rho2 = rho_2(rho_2qubit_initial)

            # Initialize combined gate history - each entry is a tuple (time_step, qubit, gate_name)
            st.session_state.level5_combined_gate_history = []
            st.session_state.level5_time_step = 0

            # Define target state
            not_gate = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, 1],
                                 [0, 0, 1, 0]])  # CNOT

            # Function for controlled rotation gate
            cr_gate_func = lambda theta: np.array([[1, 0, 0, 0],
                                                   [0, 1, 0, 0],
                                                   [0, 0, 1, 0],
                                                   [0, 0, 0, np.exp(1j * theta)]])

            # Function for parameterized swap gate
            swap_gate_func = lambda theta: np.array([
                [1, 0, 0, 0],
                [0, np.cos(theta), 1j * np.sin(theta), 0],
                [0, 1j * np.sin(theta), np.cos(theta), 0],
                [0, 0, 0, 1]
            ])

            # Calculate target state by applying gates
            state_after_not = np.dot(np.dot(not_gate, rho_2qubit_initial), not_gate.T)
            cr_with_theta = cr_gate_func(3)
            state_after_cr = np.dot(np.dot(cr_with_theta, state_after_not), cr_with_theta.T.conj())
            swap_with_theta = swap_gate_func(4.06)
            final_state = np.dot(np.dot(swap_with_theta, state_after_cr), swap_with_theta.T.conj())

            st.session_state.level5_final_state_rho = final_state
            st.session_state.level5_final_state_rho1 = rho_1(final_state)
            st.session_state.level5_final_state_rho2 = rho_2(final_state)
            st.session_state.level5_final_bloch_vector1 = bloch_vector(st.session_state.level5_final_state_rho1)
            st.session_state.level5_final_bloch_vector2 = bloch_vector(st.session_state.level5_final_state_rho2)

        # Add custom CSS for story display
        st.markdown("""
                <style>
                .story-container {
                    background-color: rgba(20, 20, 60, 0.8);
                    color:#f0e6ff;
                    padding: 20px;
                    border-radius: 10px;
                    margin: 15px 0;
                    border: 2px solid #4d79ff;
                    font-family: 'Share Tech Mono', monospace;
                    box-shadow: 0 0 15px rgba(77, 121, 255, 0.5);
                }
                .gate-info {
                    background-color: rgba(40, 10, 70, 0.7);
                    color: #f0e6ff;
                    padding: 15px;
                    border-radius: 10px;
                    margin: 15px 0;
                    border: 2px solid #9966ff;
                    font-family: 'Share Tech Mono', monospace;
                    box-shadow: 0 0 15px rgba(153, 102, 255, 0.5);
                }
                .translucent-container {
                    background-color: rgba(240, 240, 255, 0.2);
                    padding: 20px;
                    border-radius: 10px;
                    margin: 10px 0;
                }
                </style>
                """, unsafe_allow_html=True)

        # Content area
        if st.session_state.level5_story_index < len(st.session_state.level5_story_content):
            # Show story
            st.markdown(f"""
                    <div class="story-container">
                        <p style="font-size: 1.1em; line-height: 1.5;">{st.session_state.level5_story_content[st.session_state.level5_story_index]}</p>
                    </div>
                    """, unsafe_allow_html=True)

            # Continue button
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("Continue", key="level5_story_next"):
                    st.session_state.level5_story_index += 1
                    st.rerun()
        else:
            # After story is complete, show the entangled qubits interface

            # Display gate information
            st.markdown("""
                    <div class="gate-info">
                        <h3>Entanglement Gates</h3>
                        <p>Use these special gates to manipulate the entangled qubits:</p>
                    </div>
                    """, unsafe_allow_html=True)

            # Display the mathematical forms of the gates
            st.latex(r'''
                    \text{CNOT} = 
                    \begin{pmatrix} 
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 0 & 1 \\
                    0 & 0 & 1 & 0
                    \end{pmatrix}
                    \quad
                    \text{PSWAP}(\phi) = 
                    \begin{pmatrix} 
                    1 & 0 & 0 & 0 \\
                    0 & 0 & e^{i\phi} & 0 \\
                    0 & e^{i\phi} & 0 & 0 \\
                    0 & 0 & 0 & 1
                    \end{pmatrix}
                    \quad
                    \text{CR}(\theta) = 
                    \begin{pmatrix} 
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 1 & 0 \\
                    0 & 0 & 0 & e^{i\theta}
                    \end{pmatrix}
                    ''')

            # Create layout for the two qubits
            col1, col2, col3, col4 = st.columns([1, 5, 5, 1])

            with col2:
                st.header("Your Qubit")

                # Create placeholders for visualizations
                bloch_placeholder1 = st.empty()
                rho_placeholder1 = st.empty()

                # Display Bloch sphere for first qubit
                current_bloch_vector1 = bloch_vector(st.session_state.level5_state_rho1)
                with bloch_placeholder1:
                    plot_bloch_sphere(current_bloch_vector1, st.session_state.level5_final_bloch_vector1, 'blues',
                                      "Your Quantum State")

                # Gates for the first qubit
                st.write("### Apply Gates:")
                st.markdown('<div class="translucent-container">', unsafe_allow_html=True)

                # Single qubit controls
                subcol1, subcol2 = st.columns(2)
                with subcol1:
                    x_button1 = st.button('X Gate', key="x_gate_1")
                    y_button1 = st.button('Y Gate', key="y_gate_1")
                    z_button1 = st.button('Z Gate', key="z_gate_1")

                with subcol2:
                    # Two-qubit gates
                    cnot_button1 = st.button('CNOT', key="cnot_gate_1")

                    pswap_theta1 = st.slider('PSWAP φ (radians)',
                                             min_value=0.0, max_value=2 * np.pi,
                                             value=np.pi / 2, step=0.01, key="pswap_slider_1")
                    pswap_button1 = st.button('PSWAP', key="pswap_gate_1")

                    cr_theta1 = st.slider('CR θ (radians)',
                                          min_value=0.0, max_value=2 * np.pi,
                                          value=np.pi / 4, step=0.01, key="cr_slider_1")
                    cr_button1 = st.button('CR', key="cr_gate_1")

                st.markdown('</div>', unsafe_allow_html=True)

                # Display density matrix
                with rho_placeholder1:
                    display_rho(st.session_state.level5_state_rho1)

            with col3:
                st.header("Sonalika's Qubit")

                # Create placeholders for visualizations
                bloch_placeholder2 = st.empty()
                rho_placeholder2 = st.empty()

                # Display Bloch sphere for second qubit
                current_bloch_vector2 = bloch_vector(st.session_state.level5_state_rho2)
                with bloch_placeholder2:
                    plot_bloch_sphere(current_bloch_vector2, st.session_state.level5_final_bloch_vector2, 'purples',
                                      "Sonalika's Quantum State")

                # Gates for the second qubit
                st.write("### Apply Gates:")
                st.markdown('<div class="translucent-container">', unsafe_allow_html=True)

                # Single qubit controls
                subcol1, subcol2 = st.columns(2)
                with subcol1:
                    x_button2 = st.button('X Gate', key="x_gate_2")
                    y_button2 = st.button('Y Gate', key="y_gate_2")
                    z_button2 = st.button('Z Gate', key="z_gate_2")

                with subcol2:
                    # Two-qubit gates
                    cnot_button2 = st.button('CNOT', key="cnot_gate_2")

                    pswap_theta2 = st.slider('PSWAP φ (radians)',
                                             min_value=0.0, max_value=2 * np.pi,
                                             value=np.pi / 2, step=0.01, key="pswap_slider_2")
                    pswap_button2 = st.button('PSWAP', key="pswap_gate_2")

                    cr_theta2 = st.slider('CR θ (radians)',
                                          min_value=0.0, max_value=2 * np.pi,
                                          value=np.pi / 4, step=0.01, key="cr_slider_2")
                    cr_button2 = st.button('CR', key="cr_gate_2")

                st.markdown('</div>', unsafe_allow_html=True)

                # Display density matrix
                with rho_placeholder2:
                    display_rho(st.session_state.level5_state_rho2)

            # Extract timeline-based gate histories for visualization
            gate_history_q1 = []
            gate_history_q2 = []

            if len(st.session_state.level5_combined_gate_history) > 0:
                # Sort by time step to ensure proper order
                sorted_history = sorted(st.session_state.level5_combined_gate_history, key=lambda x: x[0])

                # Get the maximum time step
                max_time = sorted_history[-1][0] + 1

                # Initialize empty histories with None placeholders for each time step
                gate_history_q1 = [None] * max_time
                gate_history_q2 = [None] * max_time

                # Fill in gates at appropriate time steps
                for time, qubit, gate in sorted_history:
                    if qubit == 1:
                        gate_history_q1[time] = gate
                    elif qubit == 2:
                        gate_history_q2[time] = gate
                    elif qubit == "both":  # For two-qubit gates that affect both
                        gate_history_q1[time] = gate
                        gate_history_q2[time] = gate

                # Remove None values
                gate_history_q1 = [g for g in gate_history_q1 if g is not None]
                gate_history_q2 = [g for g in gate_history_q2 if g is not None]

            # Process button clicks for Qubit 1
            if x_button1:
                gate = pauli_x()
                st.session_state.level5_state_rho = np.kron(gate,
                                                            np.eye(2)) @ st.session_state.level5_state_rho @ np.kron(
                    gate.conj().T, np.eye(2))
                # Record gate in the combined history with time step
                st.session_state.level5_combined_gate_history.append((st.session_state.level5_time_step, 1, 'X'))
                st.session_state.level5_time_step += 1
                st.session_state.level5_state_rho1 = rho_1(st.session_state.level5_state_rho)
                st.session_state.level5_state_rho2 = rho_2(st.session_state.level5_state_rho)
                st.rerun()

            elif y_button1:
                gate = pauli_y()
                st.session_state.level5_state_rho = np.kron(gate,
                                                            np.eye(2)) @ st.session_state.level5_state_rho @ np.kron(
                    gate.conj().T, np.eye(2))
                # Record gate in the combined history with time step
                st.session_state.level5_combined_gate_history.append((st.session_state.level5_time_step, 1, 'Y'))
                st.session_state.level5_time_step += 1
                st.session_state.level5_state_rho1 = rho_1(st.session_state.level5_state_rho)
                st.session_state.level5_state_rho2 = rho_2(st.session_state.level5_state_rho)
                st.rerun()

            elif z_button1:
                gate = pauli_z()
                st.session_state.level5_state_rho = np.kron(gate,
                                                            np.eye(2)) @ st.session_state.level5_state_rho @ np.kron(
                    gate.conj().T, np.eye(2))
                # Record gate in the combined history with time step
                st.session_state.level5_combined_gate_history.append((st.session_state.level5_time_step, 1, 'Z'))
                st.session_state.level5_time_step += 1
                st.session_state.level5_state_rho1 = rho_1(st.session_state.level5_state_rho)
                st.session_state.level5_state_rho2 = rho_2(st.session_state.level5_state_rho)
                st.rerun()

            elif cnot_button1:
                gate = cnot_gate()
                st.session_state.level5_state_rho = np.dot(np.dot(gate, st.session_state.level5_state_rho), gate.T)
                # Record gate in the combined history with time step (affects both qubits)
                st.session_state.level5_combined_gate_history.append(
                    (st.session_state.level5_time_step, "both", 'CNOT'))
                st.session_state.level5_time_step += 1
                st.session_state.level5_state_rho1 = rho_1(st.session_state.level5_state_rho)
                st.session_state.level5_state_rho2 = rho_2(st.session_state.level5_state_rho)
                st.rerun()

            elif pswap_button1:
                gate = pswap_gate(pswap_theta1)
                st.session_state.level5_state_rho = np.dot(np.dot(gate, st.session_state.level5_state_rho),
                                                           gate.conj().T)
                # Record gate in the combined history with time step (affects both qubits)
                st.session_state.level5_combined_gate_history.append(
                    (st.session_state.level5_time_step, "both", f'PSWAP({pswap_theta1:.2f})'))
                st.session_state.level5_time_step += 1
                st.session_state.level5_state_rho1 = rho_1(st.session_state.level5_state_rho)
                st.session_state.level5_state_rho2 = rho_2(st.session_state.level5_state_rho)
                st.rerun()

            elif cr_button1:
                gate = cr_gate(cr_theta1)
                st.session_state.level5_state_rho = gate @ st.session_state.level5_state_rho @ gate.conj().T
                # Record gate in the combined history with time step (affects both qubits)
                st.session_state.level5_combined_gate_history.append(
                    (st.session_state.level5_time_step, "both", f'CR({cr_theta1:.2f})'))
                st.session_state.level5_time_step += 1
                st.session_state.level5_state_rho1 = rho_1(st.session_state.level5_state_rho)
                st.session_state.level5_state_rho2 = rho_2(st.session_state.level5_state_rho)
                st.rerun()

            # Process button clicks for Qubit 2
            elif x_button2:
                gate = pauli_x()
                st.session_state.level5_state_rho = np.kron(np.eye(2),
                                                            gate) @ st.session_state.level5_state_rho @ np.kron(
                    np.eye(2), gate.conj().T)
                # Record gate in the combined history with time step
                st.session_state.level5_combined_gate_history.append((st.session_state.level5_time_step, 2, 'X'))
                st.session_state.level5_time_step += 1
                st.session_state.level5_state_rho1 = rho_1(st.session_state.level5_state_rho)
                st.session_state.level5_state_rho2 = rho_2(st.session_state.level5_state_rho)
                st.rerun()

            elif y_button2:
                gate = pauli_y()
                st.session_state.level5_state_rho = np.kron(np.eye(2),
                                                            gate) @ st.session_state.level5_state_rho @ np.kron(
                    np.eye(2), gate.conj().T)
                # Record gate in the combined history with time step
                st.session_state.level5_combined_gate_history.append((st.session_state.level5_time_step, 2, 'Y'))
                st.session_state.level5_time_step += 1
                st.session_state.level5_state_rho1 = rho_1(st.session_state.level5_state_rho)
                st.session_state.level5_state_rho2 = rho_2(st.session_state.level5_state_rho)
                st.rerun()

            elif z_button2:
                gate = pauli_z()
                st.session_state.level5_state_rho = np.kron(np.eye(2),
                                                            gate) @ st.session_state.level5_state_rho @ np.kron(
                    np.eye(2), gate.conj().T)
                # Record gate in the combined history with time step
                st.session_state.level5_combined_gate_history.append((st.session_state.level5_time_step, 2, 'Z'))
                st.session_state.level5_time_step += 1
                st.session_state.level5_state_rho1 = rho_1(st.session_state.level5_state_rho)
                st.session_state.level5_state_rho2 = rho_2(st.session_state.level5_state_rho)
                st.rerun()

            elif cnot_button2:
                gate = cnot_gate()
                st.session_state.level5_state_rho = np.dot(np.dot(gate, st.session_state.level5_state_rho), gate.T)
                # Record gate in the combined history with time step (affects both qubits)
                st.session_state.level5_combined_gate_history.append(
                    (st.session_state.level5_time_step, "both", 'CNOT'))
                st.session_state.level5_time_step += 1
                st.session_state.level5_state_rho1 = rho_1(st.session_state.level5_state_rho)
                st.session_state.level5_state_rho2 = rho_2(st.session_state.level5_state_rho)
                st.rerun()

            elif pswap_button2:
                gate = pswap_gate(pswap_theta2)
                st.session_state.level5_state_rho = np.dot(np.dot(gate, st.session_state.level5_state_rho),
                                                           gate.conj().T)
                # Record gate in the combined history with time step (affects both qubits)
                st.session_state.level5_combined_gate_history.append(
                    (st.session_state.level5_time_step, "both", f'PSWAP({pswap_theta2:.2f})'))
                st.session_state.level5_time_step += 1
                st.session_state.level5_state_rho1 = rho_1(st.session_state.level5_state_rho)
                st.session_state.level5_state_rho2 = rho_2(st.session_state.level5_state_rho)
                st.rerun()


            elif cr_button2:

                gate = cr_gate(cr_theta2)

                st.session_state.level5_state_rho = gate @ st.session_state.level5_state_rho @ gate.conj().T

                # Record gate in the combined history with time step (affects both qubits)

                st.session_state.level5_combined_gate_history.append(
                    (st.session_state.level5_time_step, "both", f'CR({cr_theta2:.2f})'))

                st.session_state.level5_time_step += 1

                st.session_state.level5_state_rho1 = rho_1(st.session_state.level5_state_rho)

                st.session_state.level5_state_rho2 = rho_2(st.session_state.level5_state_rho)

                st.rerun()

                # Display circuit history

            st.markdown("### Two-Qubit Circuit History")

            # Create accurate circuit visualization based on the combined history

            max_time = st.session_state.level5_time_step

            if max_time > 0:

                # Sort history by time

                sorted_history = sorted(st.session_state.level5_combined_gate_history, key=lambda x: x[0])

                # Create the circuit figure

                circuit_fig = go.Figure()

                # Add qubit wires (length based on the number of time steps)

                circuit_fig.add_trace(go.Scatter(

                    x=[0, max_time],

                    y=[1, 1],

                    mode="lines",

                    line=dict(width=3, color="black"),

                    name="Your Qubit"

                ))

                circuit_fig.add_trace(go.Scatter(

                    x=[0, max_time],

                    y=[0, 0],

                    mode="lines",

                    line=dict(width=3, color="black"),

                    name="Sonalika's Qubit"

                ))

                # Add labels for the qubits

                circuit_fig.add_annotation(

                    x=-0.5, y=1,

                    text="Your Qubit",

                    showarrow=False,

                    font=dict(size=14, color="black")

                )

                circuit_fig.add_annotation(

                    x=-0.5, y=0,

                    text="Sonalika",

                    showarrow=False,

                    font=dict(size=14, color="black")

                )

                # Add gates at appropriate positions

                for time_step, qubit, gate_name in sorted_history:

                    if qubit == 1 or qubit == "both":
                        # Add gate for qubit 1

                        circuit_fig.add_trace(go.Scatter(

                            x=[time_step], y=[1],

                            mode="markers+text",

                            marker=dict(symbol="square", size=70, color="pink"),

                            text=[gate_name],

                            textposition='middle center',
                            textfont=dict(size=25, color="black"),

                            name=f"Gate at t={time_step}"

                        ))

                    if qubit == 2 or qubit == "both":
                        # Add gate for qubit 2

                        circuit_fig.add_trace(go.Scatter(

                            x=[time_step], y=[0],

                            mode="markers+text",

                            marker=dict(symbol="square", size=70, color="skyblue"),

                            text=[gate_name],

                            textposition='middle center',
                            textfont=dict(size=25, color="black"),

                            name=f"Gate at t={time_step}"

                        ))

                    # If it's a two-qubit gate, draw a line connecting the qubits

                    if qubit == "both":
                        circuit_fig.add_trace(go.Scatter(

                            x=[time_step, time_step],

                            y=[0, 1],

                            mode="lines",

                            line=dict(width=2, color="purple", dash="dot"),

                            name=f"Connection at t={time_step}"

                        ))

                # Update layout

                circuit_fig.update_layout(

                    xaxis=dict(range=[-1, max_time + 0.5], zeroline=False, showticklabels=True, title="Time Step"),

                    yaxis=dict(range=[-0.5, 1.5], zeroline=False, showticklabels=False),

                    margin=dict(l=30, r=30, b=30, t=30),

                    height=250,

                    width=700,

                    showlegend=False,

                    title="Two-Qubit Circuit History"

                )

                # Display the circuit

                st.plotly_chart(circuit_fig, key=f"two_qubit_circuit_{max_time}")

                # Show gate history as text for reference

                st.write("Gate sequence:")

                for time_step, qubit, gate_name in sorted_history:

                    if qubit == "both":

                        qubit_text = "Both qubits"

                    elif qubit == 1:

                        qubit_text = "Your qubit"

                    else:

                        qubit_text = "Sonalika's qubit"

                    st.write(f"Step {time_step}: {gate_name} on {qubit_text}")

            else:

                st.write("No gates applied yet. Apply gates to see the circuit history.")

            # Check if target state is reached (approx)

            target_vector1 = st.session_state.level5_final_bloch_vector1

            target_vector2 = st.session_state.level5_final_bloch_vector2

            current_vector1 = bloch_vector(st.session_state.level5_state_rho1)

            current_vector2 = bloch_vector(st.session_state.level5_state_rho2)

            distance1 = np.linalg.norm(target_vector1 - current_vector1)

            distance2 = np.linalg.norm(target_vector2 - current_vector2)

            # Level completion

            if distance1 < 0.1 and distance2 < 0.1:
                st.balloons()

                st.success(
                    "You've successfully coordinated with Sonalika to reach the target state! The entangled Bloch spheres begin to separate...")

            # Show how close they are to solution

            st.progress(max(0, 1.0 - (distance1 + distance2) / 2))

            score = 0

            if distance1 < 0.1 and distance2 < 0.1 or st.button("Click when done!"):
                score = 50

            if level(score):
                level_transition()

            st.button("End game")

    elif st.session_state.current_level == 5:
        st.markdown("""
            <h1 style="color: #FFFFFF; font-family: 'Helvetica', cursive;">
                Congratulations!
            </h1>
            <p style="font-size: 1.3em; color: #FFFFFF; line-height: 1.6; margin: 20px 0;">
                With a final quantum flash, the Bloch spheres dissolve away! You and Sonalika are free! 
                <br><br>
                The mysterious voice chuckles warmly: "Well done!" 
                <br><br>
                Suddenly, a cosmic cat wearing a tiny bow tie appears, floating beside a magnificent 
                iridescent space fish. "Hop aboard!" the Quantum Cat purrs. And of course, there's no denying a Quantum Cat and their spacefish. "
                <br><br>
            </p>
        </div>
        """, unsafe_allow_html=True)


        # Credits
        st.markdown("---")
        st.markdown("""
            Credits
    
          Game Created by: Unnati Akhouri
    
            Special thanks to
            - Sonalika Purkayastha
            - The QuantumCat
            - Sonalika's Spacefish (for their travel services)
            - All quantum adventurers who dared to escape the BlochOut!
    
            Built with Streamlit and Quantum Mechanics!
            """)

        if st.button("🔄 Play Again"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()



# Run the app
if __name__ == "__main__":
    main()
















