import plotly.graph_objects as go
import numpy as np
import streamlit as st
from qutip import *
import random
import matplotlib.pyplot as plt
#from qutip import Qobj

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


# Function to generate a random two-qubit state
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
random_state = random_two_qubit_state().full()
random_state = np.outer(random_state, np.conj(random_state))

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

#bell_phi_plus= bell_phi_plus*bell_phi_plus.dag()
##bell_phi_minus= bell_phi_minus*bell_phi_minus.dag()
#bell_psi_plus= bell_psi_plus*bell_psi_plus.dag()
#bell_psi_minus= bell_psi_minus*bell_psi_minus.dag()


    #print("\nBell States:")
    #print("Phi Plus:", bell_phi_plus)
    #print("Phi Minus:", bell_phi_minus)
    #print("Psi Plus:", bell_psi_plus)
    #print("Psi Minus:", bell_psi_minus)

    # Print W-state
w = w_state().full()
#w = Qobj(w)
w = np.outer(w, np.conj(w))


#print("\nW-State:", w)

    # Print GHZ-state
ghz = ghz_state().full()
#ghz = Qobj(ghz)
ghz = np.outer(ghz, np.conj(ghz))
#print("\nGHZ-State:", ghz)

def rho_1(rho_2qubit):
    return np.array([[rho_2qubit[0, 0] + rho_2qubit[1, 1], rho_2qubit[0, 2] + rho_2qubit[1, 3]],
                  [rho_2qubit[2, 0] + rho_2qubit[3, 1], rho_2qubit[2, 2] + rho_2qubit[3, 3]]])

# Reduced density matrix for the second qubit (tracing out the first qubit)
def rho_2(rho_2qubit):
    return np.array([[rho_2qubit[2, 2] + rho_2qubit[0, 0], rho_2qubit[2, 3] + rho_2qubit[0, 1]],
                  [rho_2qubit[3, 2] + rho_2qubit[1, 0], rho_2qubit[3, 3] + rho_2qubit[1, 1]]])


rho_2qubit_initial = random_state
#print(ghz.ptrace([1]))
    # Compute partial trace to get Bloch vectors
single_qubit_state_0 = rho_1(random_state)  # Trace out qubit 1, keep qubit 0
#print(single_qubit_state_0)
single_qubit_state_1 = rho_2(random_state)  # Trace out qubit 1, keep qubit 0
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

# Gate functions
import numpy as np

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
    # 2-qubit CNOT gate
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]])



if 'initial_state_rho1' not in st.session_state:
    st.session_state.initial_state_rho1 = single_qubit_state_0

# Initialize the first and second state vectors
if 'state_rho1' not in st.session_state:
    #st.session_state.state_vector1 = generate_random_state()

    st.session_state.state_rho1 = st.session_state.initial_state_rho1

    initial_state_rho1 = st.session_state.initial_state_rho1

    # Apply gates to the initial state to get the final state
    final_state_rho1 = apply_z_gate(apply_z_gate(apply_y_gate(apply_x_gate(initial_state_rho1))))
    final_bloch_rho1 = bloch_vector(final_state_rho1)

if 'initial_state_rho2' not in st.session_state:
    st.session_state.initial_state_rho2 = single_qubit_state_0

if 'state_rho2' not in st.session_state:
    st.session_state.state_rho2 = st.session_state.initial_state_rho2

    initial_state_rho2 = st.session_state.initial_state_rho2

    # Apply gates to the initial state to get the final state
    final_state_rho2 = apply_h_gate(apply_z_gate(apply_y_gate(apply_x_gate(initial_state_rho2))))
    final_bloch_rho2 = bloch_vector(final_state_rho2)

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
def update_system(bloch_vector, state__rho, gate_history, circuit_color, circuit_key):
    #bloch_vector[:] = state_to_bloch_vector(state_vector)
    bloch_vector
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
    rho_string = f"ρ = 0.5 (I + {r_x:.2f}σx + {r_y:.2f}σy + {r_z:.2f}σz)"
    st.write(rho_string)



# Display Bloch spheres side by side
col1, col2,col3,col4 = st.columns([5,5,5,5])

with col2:
    st.header("Qubit 1")

    # Correct condition for checking multiple keys in session_state
    if 'initial_state_rho1' not in st.session_state or 'initial_state_rho' not in st.session_state or 'initial_state_rho2' not in st.session_state:
        # Initialize the session state variables if they do not exist
        st.session_state.initial_state_rho1 = single_qubit_state_0
        st.session_state.initial_state_rho2 = single_qubit_state_1
        st.session_state.initial_state_rho = rho_2qubit_initial

    # Initialize the state_rho variables if they do not exist
    if 'state_rho1' not in st.session_state or 'state_rho2' not in st.session_state or 'state_rho' not in st.session_state:
        st.session_state.state_rho1 = st.session_state.initial_state_rho1
        st.session_state.state_rho2 = st.session_state.initial_state_rho2
        st.session_state.state_rho = st.session_state.initial_state_rho

    # Get the initial state variables
    initial_state_rho1 = st.session_state.initial_state_rho1
    initial_state_rho2 = st.session_state.initial_state_rho2
    initial_state_rho = st.session_state.initial_state_rho

    # Apply gates to the initial state to get the final state
    final_state_rho1 = rho_1(ghz)
    final_state_rho2 = rho_2(ghz)
    final_state_rho = ghz
    final_bloch_vector1 = bloch_vector(final_state_rho1)
    final_bloch_vector2 = bloch_vector(final_state_rho2)
    bloch_vector1 = bloch_vector(st.session_state.state_rho1)
    bloch_vector2 = bloch_vector(st.session_state.state_rho2)

    # Plot Bloch spheres and display the updated rho states
    plot_bloch_sphere(bloch_vector1, final_bloch_vector1, 'pinkyl', "Bloch Sphere 1")
    display_rho(st.session_state.state_rho1)

    st.write("Apply gate to Qubit 1:")
    if st.button('CNOT Gate (Qubit 1)'):
        gate=cnot_gate()
        st.session_state.state_rho =np.dot(np.dot(gate, st.session_state.state_rho), gate.T)
        st.session_state.gate_history1.append('CN')
        st.session_state.state_rho1=rho_1(st.session_state.state_rho)
        st.session_state.state_rho2=rho_2(st.session_state.state_rho)

    st.write("Apply gate to Qubit 1:")
    if st.button('X Gate (Qubit 1)'):
        gate=pauli_x()
        st.session_state.state_rho =np.kron(gate, np.eye(2)) @ st.session_state.state_rho @ np.kron(gate.conj().T, np.eye(2))
        st.session_state.gate_history1.append('X')
        st.session_state.state_rho1=rho_1(st.session_state.state_rho)
        st.session_state.state_rho2=rho_2(st.session_state.state_rho)

        #update_system(bloch_rho1, st.session_state.state_rho1, st.session_state.gate_history1, 'pink', 'circuit1')

    if st.button('Y Gate (Qubit 1)'):
        gate = pauli_y()
        st.session_state.state_rho = np.kron(gate, np.eye(2)) @ st.session_state.state_rho @ np.kron(gate.conj().T,
                                                                                                     np.eye(2))
        st.session_state.gate_history1.append('Y')
        st.session_state.state_rho1 = rho_1(st.session_state.state_rho)
        st.session_state.state_rho2=rho_2(st.session_state.state_rho)


    if st.button('Z Gate (Qubit 1)'):
        gate = pauli_z()
        st.session_state.state_rho = np.kron(gate, np.eye(2)) @ st.session_state.state_rho @ np.kron(gate.conj().T,
                                                                                                     np.eye(2))
        st.session_state.gate_history1.append('Z')
        st.session_state.state_rho1 = rho_1(st.session_state.state_rho)
        st.session_state.state_rho2=rho_2(st.session_state.state_rho)


    #if st.button('Z Gate (Qubit 1)'):
        #st.session_state.state_rho1 = apply_z_gate(st.session_state.state_rho1)
        #st.session_state.gate_history1.append('Z')
        #update_system(bloch_rho1, st.session_state.state_rho1, st.session_state.gate_history1, 'pink', 'circuit1')

    #if st.button('H Gate (Qubit 1)'):
        #st.session_state.state_rho1 = apply_h_gate(st.session_state.state_rho1)
        #st.session_state.gate_history1.append('H')
    if st.button('H Gate (Qubit 1)'):
        gate = hadamard()
        st.session_state.state_rho = np.kron(gate, np.eye(2)) @ st.session_state.state_rho @ np.kron(gate.conj().T,
                                                                                                     np.eye(2))
        st.session_state.gate_history1.append('H')
        st.session_state.state_rho1 = rho_1(st.session_state.state_rho)
        st.session_state.state_rho2=rho_2(st.session_state.state_rho)


    st.write("Apply a qubit channel:")

    px1_bar = st.slider("Rotation X angle Q1", 0.0, 1.0, 0.1)
    if st.button('Apply Rx Channel Q1'):
        gate = r_x_rotation(px1_bar)
        st.session_state.state_rho = np.kron(gate, np.eye(2)) @ st.session_state.state_rho @ np.kron(gate.conj().T,
                                                                                                     np.eye(2))
        st.session_state.gate_history1.append('Rx')
        st.session_state.state_rho1 = rho_1(st.session_state.state_rho)
        st.session_state.state_rho2=rho_2(st.session_state.state_rho)


    py1_bar = st.slider("Rotation Y angle Q1", 0.0, 1.0, 0.1)
    if st.button('Apply Ry Channel Q1'):
        gate = r_y_rotation(py1_bar)
        st.session_state.state_rho = np.kron(gate, np.eye(2)) @ st.session_state.state_rho @ np.kron(gate.conj().T,
                                                                                                     np.eye(2))
        st.session_state.gate_history1.append('Ry')
        st.session_state.state_rho1 = rho_1(st.session_state.state_rho)
        st.session_state.state_rho2=rho_2(st.session_state.state_rho)


    pz1_bar = st.slider("Rotation Z angle Q1", 0.0, 1.0, 0.1)
    if st.button('Apply Rz Channel Q1'):
        gate = r_z_rotation(pz1_bar)
        st.session_state.state_rho = np.kron(gate, np.eye(2)) @ st.session_state.state_rho @ np.kron(gate.conj().T,
                                                                                                     np.eye(2))
        st.session_state.gate_history1.append('Rz')
        st.session_state.state_rho1 = rho_1(st.session_state.state_rho)
        st.session_state.state_rho2=rho_2(st.session_state.state_rho)


    p1_damping = st.slider("Amplitude Damping Probability Q1", 0.0, 1.0, 0.1)
    if st.button('Apply Amplitude Damping Channel Q1'):
        st.session_state.state_rho1 = apply_rotation_z_gate(st.session_state.state_rho1, p1_damping)
        st.session_state.gate_history1.append(f'Amplitude Damping (p={p1_damping:.2f})')

    p1_dephasing = st.slider("Dephasing Probability Q1", 0.0, 1.0, 0.1)
    if st.button('Apply Dephasing Channel Q1'):
        st.session_state.state_rho1 = apply_rotation_z_gate(st.session_state.state_rho1, p1_dephasing)
        st.session_state.gate_history1.append(f'Dephasing (p={p1_dephasing:.2f})')
    update_system(bloch_vector1, st.session_state.state_rho1, st.session_state.gate_history1, 'pink', 'circuit1')


    print(st.session_state.state_rho)
    st.subheader("Density Matrix Real Parts")

    # Get the real part and imaginary part of the density matrix
    rho_real = np.real(st.session_state.state_rho)
    rho_imag = np.imag(st.session_state.state_rho)

    fig, (ax1) = plt.subplots(1, 1, figsize=(12, 6))

    # Plot the real part of the density matrix with a pastel color map
    cax1 = ax1.imshow(rho_real, cmap="BuPu", interpolation="nearest")
    fig.colorbar(cax1, ax=ax1)  # Add color bar to the real part heatmap
    ax1.set_title("Real Part of Density Matrix")

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the figure
    st.pyplot(fig)


with col3:
    st.header("Qubit 2")
    # Correct condition for checking multiple keys in session_state
    if 'initial_state_rho1' not in st.session_state or 'initial_state_rho' not in st.session_state or 'initial_state_rho2' not in st.session_state:
        # Initialize the session state variables if they do not exist
        st.session_state.initial_state_rho1 = single_qubit_state_0
        st.session_state.initial_state_rho2 = single_qubit_state_1
        st.session_state.initial_state_rho = rho_2qubit_initial

    # Initialize the state_rho variables if they do not exist
    if 'state_rho1' not in st.session_state or 'state_rho2' not in st.session_state or 'state_rho' not in st.session_state:
        st.session_state.state_rho1 = st.session_state.initial_state_rho1
        st.session_state.state_rho2 = st.session_state.initial_state_rho2
        st.session_state.state_rho = st.session_state.initial_state_rho

    # Get the initial state variables
    initial_state_rho1 = st.session_state.initial_state_rho1
    initial_state_rho2 = st.session_state.initial_state_rho2
    initial_state_rho = st.session_state.initial_state_rho

    # Apply gates to the initial state to get the final state
    final_state_rho1 = rho_1(ghz)
    final_state_rho2 = rho_2(ghz)
    final_state_rho = ghz
    final_bloch_vector1 = bloch_vector(final_state_rho1)
    final_bloch_vector2 = bloch_vector(final_state_rho2)
    bloch_vector1 = bloch_vector(st.session_state.state_rho1)
    bloch_vector2 = bloch_vector(st.session_state.state_rho2)

    # Plot Bloch spheres and display the updated rho states
    plot_bloch_sphere(bloch_vector2, final_bloch_vector2, 'ice', "Bloch Sphere 2")
    display_rho(st.session_state.state_rho2)

    st.write("Apply gate to Qubit 2:")
    if st.button('CNOT Gate (Qubit 2)'):
        gate = cnot_gate()
        st.session_state.state_rho = np.dot(np.dot(gate, st.session_state.state_rho), gate.T)
        st.session_state.gate_history2.append('CN')
        st.session_state.state_rho1 = rho_1(st.session_state.state_rho)
        st.session_state.state_rho2 = rho_2(st.session_state.state_rho)

    st.write("Apply gate to Qubit 2:")
    if st.button('X Gate (Qubit 2)'):
        gate = pauli_x()
        st.session_state.state_rho = np.kron(np.eye(2),gate) @ st.session_state.state_rho @ np.kron(np.eye(2),gate.conj().T)
        st.session_state.gate_history2.append('X')
        st.session_state.state_rho1 = rho_1(st.session_state.state_rho)
        st.session_state.state_rho2 = rho_2(st.session_state.state_rho)

        # update_system(bloch_rho1, st.session_state.state_rho1, st.session_state.gate_history1, 'pink', 'circuit1')

    if st.button('Y Gate (Qubit 2)'):
        gate = pauli_y()
        st.session_state.state_rho = np.kron(np.eye(2),gate) @ st.session_state.state_rho @ np.kron(np.eye(2),gate.conj().T)
        st.session_state.gate_history2.append('Y')
        st.session_state.state_rho1 = rho_1(st.session_state.state_rho)
        st.session_state.state_rho2 = rho_2(st.session_state.state_rho)

    if st.button('Z Gate (Qubit 2)'):
        gate = pauli_z()
        st.session_state.state_rho = np.kron(np.eye(2),gate) @ st.session_state.state_rho @ np.kron(np.eye(2),gate.conj().T)
        st.session_state.gate_history2.append('Z')
        st.session_state.state_rho1 = rho_1(st.session_state.state_rho)
        st.session_state.state_rho2 = rho_2(st.session_state.state_rho)

    # if st.button('Z Gate (Qubit 1)'):
    # st.session_state.state_rho1 = apply_z_gate(st.session_state.state_rho1)
    # st.session_state.gate_history1.append('Z')
    # update_system(bloch_rho1, st.session_state.state_rho1, st.session_state.gate_history1, 'pink', 'circuit1')

    # if st.button('H Gate (Qubit 1)'):
    # st.session_state.state_rho1 = apply_h_gate(st.session_state.state_rho1)
    # st.session_state.gate_history1.append('H')
    if st.button('H Gate (Qubit 2)'):
        gate = hadamard()
        st.session_state.state_rho = np.kron(np.eye(2),gate) @ st.session_state.state_rho @ np.kron(np.eye(2),gate.conj().T)
        st.session_state.gate_history2.append('H')
        st.session_state.state_rho1 = rho_1(st.session_state.state_rho)
        st.session_state.state_rho2 = rho_2(st.session_state.state_rho)

    st.write("Apply a qubit channel:")

    px2_bar = st.slider("Rotation X angle Q2", 0.0, 1.0, 0.1)
    if st.button('Apply Rx Channel Q2'):
        gate = r_x_rotation(px2_bar)
        st.session_state.state_rho = np.kron(np.eye(2),gate) @ st.session_state.state_rho @ np.kron(np.eye(2),gate.conj().T)

        st.session_state.gate_history2.append('Rx')
        st.session_state.state_rho1 = rho_1(st.session_state.state_rho)
        st.session_state.state_rho2 = rho_2(st.session_state.state_rho)

    py2_bar = st.slider("Rotation Y angle Q2", 0.0, 1.0, 0.1)
    if st.button('Apply Ry Channel Q2'):
        gate = r_y_rotation(py2_bar)
        st.session_state.state_rho = np.kron(np.eye(2),gate) @ st.session_state.state_rho @ np.kron(np.eye(2),gate.conj().T)
        st.session_state.gate_history1.append('Ry')
        st.session_state.state_rho1 = rho_1(st.session_state.state_rho)
        st.session_state.state_rho2 = rho_2(st.session_state.state_rho)

    pz2_bar = st.slider("Rotation Z angle Q2", 0.0, 1.0, 0.1)
    if st.button('Apply Rz Channel Q2'):
        gate = r_z_rotation(pz2_bar)
        st.session_state.state_rho = np.kron(np.eye(2),gate) @ st.session_state.state_rho @ np.kron(np.eye(2),gate.conj().T)

        st.session_state.gate_history1.append('Rz')
        st.session_state.state_rho1 = rho_1(st.session_state.state_rho)
        st.session_state.state_rho2 = rho_2(st.session_state.state_rho)
    p2_damping = st.slider("Amplitude Damping Probability Q2", 0.0, 1.0, 0.1)
    if st.button('Apply Amplitude Damping Channel Q2'):
        st.session_state.state_rho2 = apply_rotation_z_gate(st.session_state.state_rho1,p2_damping)
        st.session_state.gate_history2.append(f'Amplitude Damping (p={p2_damping:.2f})')


    p2_dephasing = st.slider("Dephasing Probability Q2", 0.0, 1.0, 0.1)
    if st.button('Apply Dephasing Channel Q2'):
        st.session_state.state_rho2 = apply_rotation_z_gate(st.session_state.state_rho1,p2_dephasing)
        st.session_state.gate_history2.append(f'Dephasing (p={p2_dephasing:.2f})')
    update_system(bloch_vector2, st.session_state.state_rho2, st.session_state.gate_history2, 'lightblue','circuit2')

    print(st.session_state.state_rho)
    st.subheader("Density Matrix Imaginary Parts")

    # Get the real part and imaginary part of the density matrix
    #rho_real = np.real(st.session_state.state_rho)
    rho_imag = np.imag(st.session_state.state_rho)

    fig, ( ax2) = plt.subplots(1, 1, figsize=(12, 6))

    # Plot the imaginary part of the density matrix with a pastel color map
    cax2 = ax2.imshow(rho_imag, cmap="BuPu", interpolation="nearest")
    fig.colorbar(cax2, ax=ax2)  # Add color bar to the imaginary part heatmap
    ax2.set_title("Imaginary Part of Density Matrix")

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the figure
    st.pyplot(fig)

st.write("Your task is to transform the Bloch vector to a final target state!")

