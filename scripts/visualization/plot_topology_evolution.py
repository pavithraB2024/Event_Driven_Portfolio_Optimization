
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

# Set style
sns.set(style="whitegrid")
plt.rcParams['font.family'] = 'serif'

def plot_dynamic_topology_evolution():
    # 1. Timeline Setup (t = -12 to +12)
    t = np.linspace(-12, 12, 120)
    
    # 2. Define Dynamics
    
    # A. Static Correlation Edge (AAPL <-> MSFT)
    # Stays relatively constant around 0.8 with small noise
    np.random.seed(42)
    noise = np.random.normal(0, 0.02, size=len(t))
    static_edge_weight = 0.8 + noise
    
    # B. Event-Driven Edge (Foxconn <-> AAPL)
    # Zero before t=0
    # Spikes at t=0
    # Decays after t=0
    event_edge_weight = np.zeros_like(t)
    
    decay_factor = 0.8 # Simulated Decay
    
    for i in range(len(t)):
        if t[i] < 0:
            event_edge_weight[i] = 0.0 + np.random.normal(0, 0.005) # Small noise near zero
        else:
            # Exponential decay: A * delta^t
            # We map t=0...10 to steps
            steps = t[i]
            event_edge_weight[i] = 0.9 * (decay_factor ** steps)
            
    # Applying Fusion Formula: A = (1-L)*Static + L*Event
    # Let's say we plot the RAW event signal to show the "Overwrite" potential, 
    # OR we plot the final effective weight using Lambda=0.3?
    # The prompt asks for "Event-Driven Edge" vs "Static Correlation Edge". 
    # It implies we want to see the *magnitude* of the connection.
    # If using Lambda=0.3:
    # Final_Event = 0.3 * 0.9 = 0.27
    # Final_Static = 0.7 * 0.8 = 0.56
    # This might hide the "Overwrite" effect if Static > Event visually.
    # "Overwrite" usually means the Event signal dominates the *decision*. 
    # Let's plot the RAW Magnitude of the edges to show the SIGNAL strength clearest.
    # "Line 2 (Solid Red): Event-Driven Edge... spikes heavily".
    
    # 3. Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Static Line
    ax.plot(t, static_edge_weight, color='grey', linestyle='--', linewidth=2, alpha=0.7, label='Static Correlation Edge (AAPL-MSFT)')
    
    # Event Line
    ax.plot(t, event_edge_weight, color='#d62728', linestyle='-', linewidth=3, label='Event-Driven Edge (Foxconn-AAPL)')
    
    # 4. Styling & Annotations
    ax.set_xlabel('Time (Days relative to Event)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Edge Weight Magnitude', fontsize=12, fontweight='bold')
    # ax.set_title('Figure 4: Dynamic Topology Evolution (Semantic Overwrite)', fontsize=14, fontweight='bold', pad=15)
    
    ax.axvline(x=0, color='black', linewidth=1, alpha=0.5)
    
    # Annotation Box
    text_content = "Event Detected (t=0)\n(Foxconn, AAPL, 'Delay')\nSentiment: -0.9\nSeverity: High"
    bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.9)
    ax.annotate(text_content, xy=(0, 0.9), xytext=(-6.5, 0.85),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=10, bbox=bbox_props)

    # Highlight "Overwrite Zone" if Event > Static? 
    # Or just show the spike. The text says "Overwrite forces agent...".
    
    ax.legend(loc='upper right', fontsize=11, frameon=True)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(-12, 12)
    
    # Add Shaded Region for "Event Active"
    ax.fill_between(t, 0, 1, where=(t>=0), color='#d62728', alpha=0.05, transform=ax.get_xaxis_transform())
    ax.text(5, 0.15, "Event Decay Regime", color='#d62728', alpha=0.6, fontsize=12, rotation=0, ha='center')

    plt.tight_layout()
    
    # 5. Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, "Figure_4_Topology_Evaluation.png")
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to: {output_path}")

if __name__ == "__main__":
    plot_dynamic_topology_evolution()
