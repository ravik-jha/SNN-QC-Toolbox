import streamlit as st

PASSWORD = "ravi2025"

# Initialize auth state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Show password input only if not authenticated
if not st.session_state.authenticated:
    pwd = st.text_input("Enter password", type="password")

    if pwd == PASSWORD:
        st.session_state.authenticated = True
        st.rerun()   # refresh app to hide password
    else:
        st.stop()

# 🔓 From here onward, user is authenticated
st.success("Welcome! Access granted.")


import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pennylane as qml
import random
from tqdm import tqdm
import plotly.graph_objects as go
import base64

# --- SKLEARN IMPORTS ---
from sklearn.metrics import accuracy_score as accuracy, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# --- NEUCUBE IMPORTS ---
try:
    from neucube import Reservoir
    from neucube.encoder import Delta
    from neucube.validation import Pipeline
    from neucube.sampler import SpikeCount
    from neucube.training import STDP
    from neucube.sampler.channel_sampler import ChannelContributionSampler
    from neucube.visualise import spike_raster, plot_connections
    from neucube.qfeatures import extract_features
    from neucube.qkernel import kernel_matrix, kernel
except ImportError as e:
    st.error(f"❌ Critical Import Error: {e}")
    st.info("Ensure the 'neucube' folder, 'qfeatures.py', and 'qkernel.py' are in the same directory.")
    st.stop()

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="SNN-QC", layout="wide", page_icon="🧠")

# --- SEED CONFIGURATION ---
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# --- FEATURE NAMES DEFINITION ---
# The exact order provided by user
FEATURE_NAMES = [ 
    "AF3*", "F7*", "F3*", "FC5*", "T7*", "P7*", "O1*", 
    "O2*", "P8*", "T8*", "FC6*", "F4*", "F8*", "AF4*"
]

st.title("🧠 SNN-QC: Spiking Neural Network-Quantum Computational Toolbox")
st.markdown("""
    <div style='text-align: center; color: #1E90FF; font-size: 18px; font-weight: bold;'>
        [Pipeline: Data Upload & Spike Encoding &rarr; NeuCube Spatio-temporal Learning &rarr; Spiking Feature Extraction &rarr; Quantum Kernel Classification]
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 1. SIDEBAR PARAMETERS
# ==========================================
st.sidebar.header("1. NeuCube Hyperparameters")
threshold_val = st.sidebar.slider("Delta Threshold", 0.1, 1.5, 0.8, 0.05)
res_c = st.sidebar.slider("Reservoir C", 0.1, 1.0, 0.4, 0.05)
res_l = st.sidebar.slider("Reservoir L", 0.01, 1.0, 0.169, 0.001)
stdp_pos = st.sidebar.number_input("STDP a_pos", value=0.001, format="%.4f")
stdp_neg = st.sidebar.number_input("STDP a_neg", value=-0.01, format="%.4f")
mem_thr_val = st.sidebar.number_input("Membrane Threshold", value=0.01, format="%.3f")

st.sidebar.header("2. Features Selection")
# UPDATED: Select by Name instead of Index
feat_name_1 = st.sidebar.selectbox("Feature 1 (Channel)", FEATURE_NAMES, index=0) # Defaults to AF3
feat_name_2 = st.sidebar.selectbox("Feature 2 (Channel)", FEATURE_NAMES, index=4) # Defaults to T8

# Convert Names to Indices for the Math
feat_idx_1 = FEATURE_NAMES.index(feat_name_1)
feat_idx_2 = FEATURE_NAMES.index(feat_name_2)

st.sidebar.header("3. Model Settings")
svm_c = st.sidebar.number_input("Regularization (C)", value=1.0)
k_folds = st.sidebar.slider("CV Folds", 2, 10, 5)

# --- SIDEBAR FOOTER ---
st.sidebar.markdown("---")
st.sidebar.info(
    "**© 2025 Intelligent Systems Research Centre**\n\n"
    "*Ulster University*\n\n"
    "Developed by: **Ravi Kumar Jha**\n\n"
    "Contact: Jha-R@ulster.ac.uk"
)

# ==========================================
# 2. DATA LOADING (Cached)
# ==========================================
@st.cache_data
def load_and_encode_dataset(thresh):
    """Loads CSVs and returns both RAW and ENCODED tensors"""
    base_path = './example_data/wrist_movement_eeg/'
    
    if not os.path.exists(base_path):
        # Returns: Raw, Encoded, Labels, Error Message
        return None, None, None, f"Path not found: {base_path}"

    filenameslist = ['sam'+str(idx)+'_eeg.csv' for idx in range(1,61)]
    dfs = []
    
    try:
        for filename in filenameslist:
            full_path = os.path.join(base_path, filename)
            dfs.append(pd.read_csv(full_path, header=None))
    except FileNotFoundError as e:
        return None, None, None, f"Missing File: {e}"

    fulldf = pd.concat(dfs)
    labels_path = os.path.join(base_path, 'tar_class_labels.csv')
    labels = pd.read_csv(labels_path, header=None)
    y_all = labels.values.flatten()

    # Create Raw Tensor (60 samples, 128 timepoints, 14 channels)
    X_raw = torch.tensor(fulldf.values.reshape(60, 128, 14))
    
    # Create Encoded Tensor
    encoder = Delta(threshold=thresh)
    X_encoded = encoder.encode_dataset(X_raw)
    
    return X_raw, X_encoded, y_all, None

def load_sensor_locations():
    """
    Uses eeg_mapping to find the specific XYZ coordinates.
    STRICTLY enforces 14 points to match FEATURE_NAMES.
    """
    base_path = './example_data/wrist_movement_eeg/'
    coord_path = os.path.join(base_path, 'brain_coordinates.csv')
    map_path = os.path.join(base_path, 'eeg_mapping.csv')
    
    if os.path.exists(coord_path) and os.path.exists(map_path):
        # 1. Load all reservoir coordinates (e.g., 1471 points)
        all_coords = pd.read_csv(coord_path, header=None).values
        
        # 2. Load the mapping indices
        mapping_indices = pd.read_csv(map_path, header=None).values.flatten().astype(int)
        
        # --- FIX 1: FORCE EXACTLY 14 INDICES ---
        # If the file has 15, we take the first 14. 
        # If the file has 14, this does nothing (safe).
        if len(mapping_indices) > 14:
            mapping_indices = mapping_indices[:14]
            
        # 3. Select the rows
        channel_coords = all_coords[mapping_indices]
        
        return channel_coords
    else:
        return None

# ==========================================
# 3. MAIN EXECUTION FLOW
# ==========================================

st.markdown("""
    <h2 style='font-size: 24px;'> Data Upload & Spike Encoding </h2>
""", unsafe_allow_html=True)


if st.button("Load & Encode Data"):
    with st.spinner(f"Loading files and applying Delta Encoding (Thresh={threshold_val})..."):
        X_raw, X_encoded, y_data, err = load_and_encode_dataset(threshold_val)
        
        if err:
            st.error(err)
        else:
            st.success(f"Data Loaded Successfully!")
            
            col1, col2 = st.columns(2)
            
            # --- CUSTOM FONT SIZE METRICS ---
            # using HTML to control the size (e.g., 20px is smaller than the default metric)
            with col1:
                st.markdown(f"""
                <div style="background-color: #262730; padding: 10px; border-radius: 5px;">
                    <p style="margin:0; font-size: 20px; color: #9da3a8;">Input Shape</p>
                    <p style="margin:0; font-size: 20px; font-weight: bold; color: white;">{X_encoded.shape}</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                st.markdown(f"""
                <div style="background-color: #262730; padding: 10px; border-radius: 5px;">
                    <p style="margin:0; font-size: 20px; color: #9da3a8;">Total Labels</p>
                    <p style="margin:0; font-size: 20px; font-weight: bold; color: white;">{len(y_data)}</p>
                </div>
                """, unsafe_allow_html=True)
            # --------------------------------
            
            # Save to session state
            st.session_state['X_raw'] = X_raw
            st.session_state['X'] = X_encoded 
            st.session_state['y'] = y_data
            st.session_state['data_ready'] = True
            
            # Save RAW and ENCODED to session state
            st.session_state['X_raw'] = X_raw
            st.session_state['X'] = X_encoded 
            st.session_state['y'] = y_data
            st.session_state['data_ready'] = True

# Check if data is ready
if st.session_state.get('data_ready', False):
    X_raw = st.session_state['X_raw']
    X = st.session_state['X']
    y = st.session_state['y']
    
        # --- UPDATED: VISUALIZATION SECTION ---
    with st.expander("Raw Signals & Spikes", expanded=True):
        st.caption("Encode Spikes")
        
        # --- SELECTORS ---
        col_viz_1, col_viz_2 = st.columns(2)
        
        with col_viz_1:
            # 1. Select Sample
            sel_sample = st.slider("Select Trial", 0, len(X)-1, 0)
            
        with col_viz_2:
            # 2. Select Feature (Channel) - THIS IS THE NEW DROP DOWN
            sel_channel = st.selectbox("Select Feature (Channel)", FEATURE_NAMES, index=0, key="viz_chan_sel")
        
        # Convert the selected Name (e.g., "T7") to Index (e.g., 4)
        ch_idx_viz = FEATURE_NAMES.index(sel_channel)
        
        # --- PREPARE DATA FOR PLOTTING ---
        # Raw Data (Continuous values)
        raw_sig = X_raw[sel_sample, :, ch_idx_viz].numpy()
        
        # Encoded Data (Discrete Spikes: -1, 0, 1)
        spike_sig = X[sel_sample, :, ch_idx_viz].numpy()
        
        # --- PLOTTING LOGIC ---
        # Create two subplots sharing the X-axis (Time)
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 3.5), gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot 1: Raw Signal
        ax1.plot(raw_sig, color='#1E90FF', linewidth=1.0, label='Raw Input')
        ax1.set_ylabel(" Signal Values")
        ax1.set_title(f"Feature {sel_channel} (Trial {sel_sample})")
        ax1.grid(True, alpha=0.5)
        ax1.legend(loc='upper right')
        
        # Plot 2: Spikes 
        # Using a stem plot to clearly show discrete events
        markerline, stemlines, baseline = ax2.stem(
            np.arange(len(spike_sig)), 
            spike_sig, 
            linefmt='#1E90FF', 
            markerfmt=' ', # You can change 'o' to ' ' if you want bars only (no dots)
            basefmt='gray'
        )
        
        # Style the lines to be thin and clean
        plt.setp(stemlines, 'linewidth', 1.2)
        #plt.setp(markerline, 'markersize', 2)
        plt.setp(baseline, 'linewidth', 0.1, 'alpha', 0.1)
        
        ax2.set_ylabel("Spike State")
        ax2.set_xlabel("Time Steps")
        
        # --- KEY CHANGES HERE ---
        ax2.set_yticks([])   # Only show 0 and 1 on the axis
        #ax2.set_ylim(-0.1, 1.25) # Crop out the negative space (-1) completely
        # ------------------------
        
        ax2.set_title("Output Spikes") 
        #ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)

    st.divider()
    
    #st.header("NeuCube Spatio-temporal Learning")
    st.markdown("""
        <h2 style='font-size: 24px;'> NeuCube Spatio-temporal Learning </h2>
    """, unsafe_allow_html=True)
    
    #SENSOR VISUALIZATION
    # Define Standard 10-20 Coordinates
    if st.button("NeuCube Initialisation & Mapping"):
        st.session_state['map_initialised'] = True
    # Render the Map only if initialized
    if st.session_state.get('map_initialised', False):
        
        # --- DATA PREPARATION ---
        STANDARD_10_20 = {
            "AF3": [-30, 50, 30], "O2": [20, -90, 10],
            "F7": [-50, 30, 0], "P8": [60, -60, 0],
            "F3": [-40, 30, 40], "T8": [70, -20, 0],
            "FC5": [-60, 0, 30], "FC6": [60, 0, 30],
            "T7": [-70, -20, 0], "F4":  [40, 30, 40],
            "P7": [-60, -60, 0], "F8": [50, 30, 0],
            "O1": [-20, -90, 10], "AF4": [30, 50, 30]
        }

        plot_coords = []
        plot_names = []
        
        for raw_name in FEATURE_NAMES:
            clean_name = raw_name.replace('*', '') 
            if clean_name in STANDARD_10_20:
                plot_coords.append(STANDARD_10_20[clean_name])
                plot_names.append(raw_name)
            else:
                plot_coords.append([0,0,0])
                plot_names.append(raw_name)
                
        plot_coords = np.array(plot_coords)

        # --- VIEW CONTROL ---
        with st.expander("Schematic Brain Template", expanded=True):
            col_ctrl_1, col_ctrl_2 = st.columns([1, 4])
            with col_ctrl_1:
                clean_view = st.toggle("Grid & Axes", value=True)

            # --- PLOTTING ---
            fig_3d = go.Figure()

            # Electrodes
            fig_3d.add_trace(go.Scatter3d(
                x=plot_coords[:, 0], y=plot_coords[:, 1], z=plot_coords[:, 2],
                mode='markers+text',
                text=plot_names,
                textposition="top center",
                textfont=dict(family="Arial Black", size=12, color="white"), 
                marker=dict(size=12, color='#FF4B4B', opacity=1.0, line=dict(width=2, color='white')),
                name="Electrodes"
            ))

            # Ghost Head Model
            phi = np.linspace(0, 2*np.pi, 20)
            theta = np.linspace(0, np.pi, 10)
            phi, theta = np.meshgrid(phi, theta)
            r = 90 
            x_sphere = r * np.sin(theta) * np.cos(phi)
            y_sphere = r * np.sin(theta) * np.sin(phi)
            z_sphere = r * np.cos(theta) - 10 
        
            fig_3d.add_trace(go.Mesh3d(
                x=x_sphere.flatten(), y=y_sphere.flatten(), z=z_sphere.flatten(),
                color='gray', opacity=0.2, name='Head Model', alphahull=0 
            ))

            # Layout Logic
            if clean_view:
                grid_status = False
                axis_range = [-100, 100]
            else:
                grid_status = True
                axis_range = None

            fig_3d.update_layout(
                title="Schematic Head Model Visualisation",
                scene=dict(
                    xaxis=dict(range=axis_range, showgrid=grid_status, zeroline=grid_status, showticklabels=grid_status, title='X' if grid_status else ''),
                    yaxis=dict(range=axis_range, showgrid=grid_status, zeroline=grid_status, showticklabels=grid_status, title='Y' if grid_status else ''),
                    zaxis=dict(range=axis_range, showgrid=grid_status, zeroline=grid_status, showticklabels=grid_status, title='Z' if grid_status else ''),
                    aspectmode='cube'
                ),
                margin=dict(l=0, r=0, b=0, t=40),
                height=700,
                showlegend=False
            )

            st.plotly_chart(fig_3d, use_container_width=True)


    # ==========================================
    # B. SIMULATION LOGIC (COMES SECOND)
    # ==========================================
    
    # We only show the Run Simulation button if the map has been initialized
    if st.session_state.get('map_initialised', False):
        st.divider() # Visual separation
        
        col_sim_1, col_sim_2 = st.columns([1, 3])
        
        with col_sim_1:
            # Add some vertical padding so the button aligns nicely
            st.write("") 
            st.write("")
            run_sim = st.button("▶ Run NeuCube Simulation")

        if run_sim:
            with col_sim_2:
                # 1. Create a placeholder for the "Thinking Brain" animation
                brain_placeholder = st.empty()
                
                # 2. Display an animated GIF in the placeholder
                # Updated parameter: use_container_width instead of use_column_width
                try:
                    file_path = "brain_activity.gif" # Your filename
                    with open(file_path, "rb") as f:
                        contents = f.read()
                        data_url = base64.b64encode(contents).decode("utf-8")

                    # Inject HTML to force the GIF to play
                    brain_placeholder.markdown(
                        f'<img src="data:image/gif;base64,{data_url}" width="300" style="border-radius: 10px;">',
                        unsafe_allow_html=True,
                    )
                except FileNotFoundError:
                    brain_placeholder.info("🧠 NeuCube is thinking... (GIF not found)")

                # 3. Run the heavy computation
                with st.spinner("Spatio-temporal Training..."):
                    
                    # Initialize Reservoir
                    res = Reservoir(inputs=X.shape[2], c=res_c, l=res_l)
                    learning_rule = STDP(a_pos=stdp_pos, a_neg=stdp_neg)
                    sam = SpikeCount()

                    # Simulate
                    s_act_all = res.simulate(
                        X,
                        train=True,                  
                        learning_rule=learning_rule,
                        mem_thr=mem_thr_val,                
                        refractory_period=5,
                        verbose=True 
                    )

                    # B. Extraction (Step 3 Logic - done here to save state)
                    state_vectors_all = sam.sample(s_act_all)
                    snn_features = extract_features(state_vectors_all, res.w_in)
                    
                    # Save everything
                    st.session_state['snn_features'] = snn_features
                    st.session_state['features_ready'] = True
                
                brain_placeholder.empty()
            
            st.success("Simulation Complete!")


    # ==========================================
    # STEP 3: FEATURES EXTRACTION (Visual Section)
    # ==========================================
    
    # This block is UN-INDENTED so it sits outside the button logic
    if st.session_state.get('features_ready', False):
        st.divider()
        #st.header("Spiking Feature Extraction")
        st.markdown("""
                <h2 style='font-size: 24px;'> Spiking Feature Extraction </h2>
        """, unsafe_allow_html=True)
        st.caption("Trained spike frequency state vectors.")
        
        # Display the Shape (The proof that Step 3 is done)
        st.write(f"**Extracted Features Shape:** {st.session_state['snn_features'].shape}")
        
        # Optional: Add a plot here if you want to visualize the features
        ##with st.expander("Inspect Feature Vector"):
            ##st.bar_chart(st.session_state['snn_features'][0])
             

    # Check if features are ready
    if st.session_state.get('features_ready', False):
        snn_features = st.session_state['snn_features']
        
        st.divider()
        #st.header("Quantum Kernel Classification")
        st.markdown("""
                <h2 style='font-size: 24px;'> Quantum Kernel Classification </h2>
        """, unsafe_allow_html=True)
        
        # 1. Prepare Data
        class_mask = np.isin(y, [1, 2])
        binary_class = snn_features[class_mask]
        y_final = y[class_mask]
        
        # Select Features based on Indices derived from Names
        feature_indices = [int(feat_idx_1), int(feat_idx_2)]
        X_final = binary_class[:, feature_indices]
        
        col1, col2 = st.columns(2)
        col1.write(f"**Quantum Kernel Input Shape:** {X_final.shape}")
        
        # Show Names in UI
        col2.write(f"**Selected Features:** {feat_name_1}, {feat_name_2} ")

        # 2. Visualise Quantum Circuit
        with st.expander("View Quantum Kernel Circuit"):
            try:
                fig, ax = qml.draw_mpl(kernel)(X_final[0], X_final[1])
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Circuit visualization error: {e}")

        # 3. Quantum Kernel Classification
        if st.button("Run Cross-Validation"):
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=int(SEED))
            y_total2, pred_total2 = [], []

            svm = SVC(kernel=kernel_matrix, C=svm_c)

            progress_bar = st.progress(0)
            status_txt = st.empty()

            for i, (train_index, test_index) in enumerate(kf.split(X_final)):
                status_txt.text(f"Running Fold {i+1}/{k_folds}...")
                
                X_train, X_test = X_final[train_index], X_final[test_index]
                y_train, y_test = y_final[train_index], y_final[test_index]

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                svm.fit(X_train, y_train)
                pred2 = svm.predict(X_test)

                y_total2.extend(y_test)
                pred_total2.extend(pred2)
                
                progress_bar.progress((i + 1) / k_folds)

            status_txt.text("Validation Complete.")
            
            # --- FINAL METRICS ---
            final_acc = accuracy(y_total2, pred_total2)
            st.success(f"Final Accuracy: {final_acc:.2%}")
            
            # --- CONFUSION MATRIX ---
            st.markdown("##### Confusion Matrix")   # smaller than subheader
            
            cm = confusion_matrix(y_total2, pred_total2)
            unique_labels = sorted(list(set(y_total2)))
            
            col_cm_1, col_cm_2 = st.columns([1, 2])
            
            with col_cm_1:
                fig_cm, ax_cm = plt.subplots(figsize=(4, 2))
                
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
                disp.plot(cmap='Blues', ax=ax_cm, colorbar=False)
                
                ax_cm.set_title("Confusion Matrix", fontsize=6)
                ax_cm.set_xlabel("Predicted label", fontsize=6)
                ax_cm.set_ylabel("True label", fontsize=6)
                ax_cm.tick_params(axis='both', labelsize=6)
                
                st.pyplot(fig_cm, width='content')
                
                
st.divider()

with st.expander("📂 Data & Material"):
    st.markdown("""
    :blue[👉 **Data Availability:**]
    The EEG dataset and NeuCube software environment are kindly made available from the Auckland University of Technology at: [https://kedri.aut.ac.nz/neucube].
 
    :blue[👉 **Code Availability:**]
    The complete source code for the **SNN-QC** toolbox is kindly made available in the GitHub at: [https://github.com/ravik-jha/SNN-QC].  
    """)                
               
# --- REFERENCES SECTION ---
st.divider()

with st.expander("📚 References & Acknowledgement"):
    st.markdown("""
    :blue[**References**]
    
    [1]. Jha, R. K., Kasabov, N., Bhattacharyya, S., Coyle, D., & Prasad, G. (2025). A hybrid spiking neural network-quantum framework for spatio-temporal data classification: a case study on EEG data. 
      EPJ Quantum Technology, 12(1), 1-23. [https://doi.org/10.1140/epjqt/s40507-025-00443-1]
      
    [2]. Kasabov, N. (2014). NeuCube: A spiking neural network architecture for mapping, learning and understanding of spatio-temporal brain data. Neural networks, 52, 62-76. [https://doi.org/10.1016/j.neunet.2014.01.006] 
    
    :blue[**Acknowledgement**]
    
    This toolbox demonstration was developed as part of the research conducted at the Intelligent Systems Research Centre, Ulster University. 
    It is intended solely for academic research and demonstration purposes. This is a foundational prototype; all rights regarding future development and commercial 
    utilization are reserved and subject to copyright.
    """)