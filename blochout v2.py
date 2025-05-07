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
    final_state_rho1 = rho_1(w)
    final_state_rho2 = rho_2(w)
    final_state_rho = ghz
    final_bloch_vector1 = bloch_vector(final_state_rho1)
    final_bloch_vector2 = bloch_vector(final_state_rho2)
    bloch_vector1 = bloch_vector(st.session_state.state_rho1)
    bloch_vector2 = bloch_vector(st.session_state.state_rho2)

    # Plot Bloch spheres and display the updated rho states
    plot_bloch_sphere(bloch_vector1, final_bloch_vector1, 'blues', "Bloch Sphere 1")
    display_rho(st.session_state.state_rho1)

    st.write("Apply gate to Qubit 1:")
    st.markdown('<div class="translucent-container">', unsafe_allow_html=True)
    subcol1, subcol2, subcol3, subcol4 = st.columns(4)
    with subcol1:
        if st.button('CNOT Gate (Qubit 1)'):
            gate=cnot_gate()
            st.session_state.state_rho =np.dot(np.dot(gate, st.session_state.state_rho), gate.T)
            st.session_state.gate_history1.append('CN')
            st.session_state.state_rho1=rho_1(st.session_state.state_rho)
            st.session_state.state_rho2=rho_2(st.session_state.state_rho)

        # Add slider to choose theta for PSWAP
        theta = st.slider('Theta for PSWAP (radians)', min_value=0.0, max_value=2 * np.pi, value=np.pi / 2, step=0.01)

        # Button to apply PSWAP Gate
        if st.button('PSWAP Gate'):
            gate = pswap_gate(theta)
            st.session_state.state_rho = np.dot(np.dot(gate, st.session_state.state_rho), gate.conj().T)
            st.session_state.gate_history1.append(f'PSWAP({theta:.2f})')
            st.session_state.state_rho1 = rho_1(st.session_state.state_rho)
            st.session_state.state_rho2 = rho_2(st.session_state.state_rho)

        if st.button('CZ Gate'):
            gate = cz_gate()
            st.session_state.state_rho = gate @ st.session_state.state_rho @ gate.conj().T
            st.session_state.gate_history1.append('CZ')
            st.session_state.state_rho1 = rho_1(st.session_state.state_rho)
            st.session_state.state_rho2 = rho_2(st.session_state.state_rho)

        # CR Gate with theta slider
        theta = st.slider('CR Gate θ (radians)', min_value=0.0, max_value=2 * np.pi, value=np.pi / 4, step=0.01)
        if st.button('CR Gate'):
            gate = cr_gate(theta)
            st.session_state.state_rho = gate @ st.session_state.state_rho @ gate.conj().T
            st.session_state.gate_history1.append(f'CR({round(theta, 2)})')
            st.session_state.state_rho1 = rho_1(st.session_state.state_rho)
            st.session_state.state_rho2 = rho_2(st.session_state.state_rho)

        #update_system(bloch_rho1, st.session_state.state_rho1, st.session_state.gate_history1, 'pink', 'circuit1')
    with subcol2:
        if st.button('X Gate (Qubit 1)'):
            gate=pauli_x()
            st.session_state.state_rho =np.kron(gate, np.eye(2)) @ st.session_state.state_rho @ np.kron(gate.conj().T, np.eye(2))
            st.session_state.gate_history1.append('X')
            st.session_state.state_rho1=rho_1(st.session_state.state_rho)
            st.session_state.state_rho2=rho_2(st.session_state.state_rho)

        if st.button('Y Gate (Qubit 1)'):
            gate = pauli_y()
            st.session_state.state_rho = np.kron(gate, np.eye(2)) @ st.session_state.state_rho @ np.kron(gate.conj().T,np.eye(2))

            st.session_state.gate_history1.append('Y')
            st.session_state.state_rho1 = rho_1(st.session_state.state_rho)
            st.session_state.state_rho2=rho_2(st.session_state.state_rho)

        if st.button('Z Gate (Qubit 1)'):
            gate = pauli_z()
            st.session_state.state_rho = np.kron(gate, np.eye(2)) @ st.session_state.state_rho @ np.kron(gate.conj().T,np.eye(2))

            st.session_state.gate_history1.append('Z')
            st.session_state.state_rho1 = rho_1(st.session_state.state_rho)
            st.session_state.state_rho2=rho_2(st.session_state.state_rho)

        if st.button('H Gate (Qubit 1)'):
            gate = hadamard()
            st.session_state.state_rho = np.kron(gate, np.eye(2)) @ st.session_state.state_rho @ np.kron(gate.conj().T,np.eye(2))

            st.session_state.gate_history1.append('H')
            st.session_state.state_rho1 = rho_1(st.session_state.state_rho)
            st.session_state.state_rho2=rho_2(st.session_state.state_rho)

        if st.button('S Gate (Qubit 1)'):
            gate = s_gate()
            st.session_state.state_rho = np.kron(gate, np.eye(2)) @ st.session_state.state_rho @ np.kron(gate.conj().T,
                                                                                                         np.eye(2))
            st.session_state.gate_history1.append('S')
            st.session_state.state_rho1 = rho_1(st.session_state.state_rho)
            st.session_state.state_rho2 = rho_2(st.session_state.state_rho)

        # T Gate for Qubit 1
        if st.button('T Gate (Qubit 1)'):
            gate = t_gate()
            st.session_state.state_rho = np.kron(gate, np.eye(2)) @ st.session_state.state_rho @ np.kron(gate.conj().T,
                                                                                                         np.eye(2))
            st.session_state.gate_history1.append('T')
            st.session_state.state_rho1 = rho_1(st.session_state.state_rho)
            st.session_state.state_rho2 = rho_2(st.session_state.state_rho)
    with subcol3:
        st.write("Apply a qubit channel:")

        px1_bar = st.slider("Rotation X angle Q1", 0.0, 1.0, 0.1)
        if st.button('Apply Rx Channel Q1'):
            gate = r_x_rotation(px1_bar)
            st.session_state.state_rho = np.kron(gate, np.eye(2)) @ st.session_state.state_rho @ np.kron(gate.conj().T,np.eye(2))

            st.session_state.gate_history1.append('Rx')
            st.session_state.state_rho1 = rho_1(st.session_state.state_rho)
            st.session_state.state_rho2=rho_2(st.session_state.state_rho)


        py1_bar = st.slider("Rotation Y angle Q1", 0.0, 1.0, 0.1)
        if st.button('Apply Ry Channel Q1'):
            gate = r_y_rotation(py1_bar)
            st.session_state.state_rho = np.kron(gate, np.eye(2)) @ st.session_state.state_rho @ np.kron(gate.conj().T,np.eye(2))

            st.session_state.gate_history1.append('Ry')
            st.session_state.state_rho1 = rho_1(st.session_state.state_rho)
            st.session_state.state_rho2=rho_2(st.session_state.state_rho)


        pz1_bar = st.slider("Rotation Z angle Q1", 0.0, 1.0, 0.1)
        if st.button('Apply Rz Channel Q1'):
            gate = r_z_rotation(pz1_bar)
            st.session_state.state_rho = np.kron(gate, np.eye(2)) @ st.session_state.state_rho @ np.kron(gate.conj().T,np.eye(2))

            st.session_state.gate_history1.append('Rz')
            st.session_state.state_rho1 = rho_1(st.session_state.state_rho)
            st.session_state.state_rho2=rho_2(st.session_state.state_rho)

    with subcol4:
        p1_damping = st.slider("Amplitude Damping Probability Q1", 0.0, 1.0, 0.1)
        if st.button('Apply Amplitude Damping Channel Q1'):
            st.session_state.state_rho1 = apply_rotation_z_gate(st.session_state.state_rho1, p1_damping)
            st.session_state.gate_history1.append(f'Amplitude Damping (p={p1_damping:.2f})')

        p1_dephasing = st.slider("Dephasing Probability Q1", 0.0, 1.0, 0.1)
        if st.button('Apply Dephasing Channel Q1'):
            st.session_state.state_rho1 = apply_rotation_z_gate(st.session_state.state_rho1, p1_dephasing)
            st.session_state.gate_history1.append(f'Dephasing (p={p1_dephasing:.2f})')
        update_system(bloch_vector1, st.session_state.state_rho1, st.session_state.gate_history1, 'pink', 'circuit1')

    st.markdown('</div>', unsafe_allow_html=True)
    #print(st.session_state.state_rho)
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
    plot_bloch_sphere(bloch_vector2, final_bloch_vector2, 'purples', "Bloch Sphere 2")
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

